//! Typed document attributes and borrowed, allocation-free filter expressions.
//! Metadata is separate from document text and is never embedded or searched.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize};

/// An owned set of document attributes. Field names are case-sensitive.
/// Inserting an existing name replaces its value, even if its type changes.
///
/// ```
/// use kestrel::{Field, Metadata};
///
/// const TITLE: Field<'static, String> = Field::new("title");
/// let mut metadata = Metadata::new();
/// metadata.insert(TITLE, "Rust guide");
/// assert_eq!(metadata.get(TITLE).map(String::as_str), Some("Rust guide"));
/// ```
///
/// A field's type determines which values can be inserted:
///
/// ```compile_fail
/// use kestrel::{Field, Metadata};
///
/// let mut metadata = Metadata::new();
/// metadata.insert(Field::<i64>::new("year"), "2025");
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Metadata {
    values: BTreeMap<String, MetadataValue>,
}

/// The supported owned representations, exposed for the sealed type trait.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Bool(bool),
    Integer(i64),
    Number(FiniteF64),
    Strings(Vec<String>),
}

mod sealed {
    pub trait Sealed {}
}

/// A supported metadata type: `String`, `bool`, `i64`, [`FiniteF64`], or
/// `Vec<String>`. This trait is sealed and cannot be implemented externally.
pub trait MetadataType: sealed::Sealed + Sized {
    #[doc(hidden)]
    fn into_value(self) -> MetadataValue;

    #[doc(hidden)]
    fn as_value(value: &MetadataValue) -> Option<&Self>;
}

macro_rules! metadata_type {
    ($ty:ty, $variant:ident) => {
        impl sealed::Sealed for $ty {}

        impl MetadataType for $ty {
            fn into_value(self) -> MetadataValue {
                MetadataValue::$variant(self)
            }

            fn as_value(value: &MetadataValue) -> Option<&Self> {
                match value {
                    MetadataValue::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    };
}

metadata_type!(String, String);
metadata_type!(bool, Bool);
metadata_type!(i64, Integer);
metadata_type!(FiniteF64, Number);
metadata_type!(Vec<String>, Strings);

impl Metadata {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value, allocating a field name only when the name is new.
    pub fn insert<T: MetadataType>(&mut self, field: Field<'_, T>, value: impl Into<T>) {
        let value = value.into().into_value();
        if let Some(previous) = self.values.get_mut(field.name) {
            *previous = value;
        } else {
            self.values.insert(field.name.to_owned(), value);
        }
    }

    /// Borrow a value, or return `None` if the name is absent or has another type.
    pub fn get<T: MetadataType>(&self, field: Field<'_, T>) -> Option<&T> {
        self.values.get(field.name).and_then(T::as_value)
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
}

/// A finite floating-point metadata value. NaN and both infinities are rejected
/// by construction and deserialization. Comparisons use ordinary numeric
/// equality and ordering, including `-0.0 == 0.0`.
///
/// Raw `f64` values must be explicitly validated before insertion:
///
/// ```compile_fail
/// use kestrel::{Field, FiniteF64, Metadata};
///
/// let mut metadata = Metadata::new();
/// metadata.insert(Field::<FiniteF64>::new("rating"), 4.5_f64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(try_from = "f64")]
pub struct FiniteF64(f64);

impl FiniteF64 {
    pub const fn new(value: f64) -> Option<Self> {
        if value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> f64 {
        self.0
    }
}

impl TryFrom<f64> for FiniteF64 {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value).ok_or("expected a finite f64")
    }
}

impl From<FiniteF64> for f64 {
    fn from(value: FiniteF64) -> Self {
        value.get()
    }
}

/// A borrowed field name with a compile-time value type. Copying a field never
/// copies its name and does not require `T: Copy` or `T: Clone`.
///
/// Only numeric fields provide ranges:
///
/// ```compile_fail
/// use kestrel::Field;
///
/// let title = Field::<String>::new("title");
/// let _ = title.range(1_i64..10);
/// ```
#[derive(Debug)]
pub struct Field<'a, T> {
    name: &'a str,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for Field<'_, T> {}

impl<T> Clone for Field<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T> Field<'a, T> {
    pub const fn new(name: &'a str) -> Self {
        Self {
            name,
            marker: PhantomData,
        }
    }

    pub const fn name(&self) -> &'a str {
        self.name
    }

    /// Check name presence, regardless of the stored value's type.
    pub fn exists(self) -> Filter<'a> {
        Filter {
            kind: FilterKind::Exists(self.name),
        }
    }
}

impl<'a> Field<'a, String> {
    pub fn eq(self, value: &'a str) -> Filter<'a> {
        Filter {
            kind: FilterKind::StringEq(self, value),
        }
    }

    pub fn in_set(self, values: &'a [&'a str]) -> Filter<'a> {
        Filter {
            kind: FilterKind::StringIn(self, values),
        }
    }
}

impl<'a> Field<'a, bool> {
    pub fn eq(self, value: bool) -> Filter<'a> {
        Filter {
            kind: FilterKind::BoolEq(self, value),
        }
    }
}

impl<'a> Field<'a, i64> {
    pub fn eq(self, value: i64) -> Filter<'a> {
        Filter {
            kind: FilterKind::IntegerEq(self, value),
        }
    }

    pub fn in_set(self, values: &'a [i64]) -> Filter<'a> {
        Filter {
            kind: FilterKind::IntegerIn(self, values),
        }
    }

    /// Preserve the bounds exactly, without adjusting endpoints arithmetically.
    /// Empty or reversed ranges match nothing.
    pub fn range(self, range: impl RangeBounds<i64>) -> Filter<'a> {
        Filter {
            kind: FilterKind::IntegerRange(
                self,
                (range.start_bound().cloned(), range.end_bound().cloned()),
            ),
        }
    }
}

impl<'a> Field<'a, FiniteF64> {
    pub fn eq(self, value: FiniteF64) -> Filter<'a> {
        Filter {
            kind: FilterKind::NumberEq(self, value),
        }
    }

    pub fn in_set(self, values: &'a [FiniteF64]) -> Filter<'a> {
        Filter {
            kind: FilterKind::NumberIn(self, values),
        }
    }

    /// Preserve the bounds exactly. Empty or reversed ranges match nothing.
    pub fn range(self, range: impl RangeBounds<FiniteF64>) -> Filter<'a> {
        Filter {
            kind: FilterKind::NumberRange(
                self,
                (range.start_bound().cloned(), range.end_bound().cloned()),
            ),
        }
    }
}

impl<'a> Field<'a, Vec<String>> {
    pub fn contains(self, value: &'a str) -> Filter<'a> {
        Filter {
            kind: FilterKind::StringsContains(self, value),
        }
    }
}

/// A borrowed filter expression. Construction and evaluation allocate nothing.
/// String comparisons are exact and case-sensitive; integers and numbers are
/// never coerced into one another. Missing or differently typed values fail
/// leaf comparisons. Negation is their logical complement.
///
/// ```
/// use kestrel::{Field, Filter, Metadata};
///
/// const YEAR: Field<'static, i64> = Field::new("year");
/// const TAGS: Field<'static, Vec<String>> = Field::new("tags");
/// let mut metadata = Metadata::new();
/// metadata.insert(YEAR, 2025_i64);
/// metadata.insert(TAGS, vec!["rust".to_owned()]);
///
/// let clauses = [YEAR.range(2020..=2025), TAGS.contains("rust")];
/// let filter = Filter::and(&clauses);
/// assert!(filter.matches(&metadata));
/// assert!(!filter.negate().matches(&metadata));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Filter<'a> {
    kind: FilterKind<'a>,
}

#[derive(Clone, Copy, Debug)]
enum FilterKind<'a> {
    And(&'a [Filter<'a>]),
    Or(&'a [Filter<'a>]),
    Not(&'a Filter<'a>),
    Exists(&'a str),
    StringEq(Field<'a, String>, &'a str),
    StringIn(Field<'a, String>, &'a [&'a str]),
    BoolEq(Field<'a, bool>, bool),
    IntegerEq(Field<'a, i64>, i64),
    IntegerIn(Field<'a, i64>, &'a [i64]),
    IntegerRange(Field<'a, i64>, (Bound<i64>, Bound<i64>)),
    NumberEq(Field<'a, FiniteF64>, FiniteF64),
    NumberIn(Field<'a, FiniteF64>, &'a [FiniteF64]),
    NumberRange(Field<'a, FiniteF64>, (Bound<FiniteF64>, Bound<FiniteF64>)),
    StringsContains(Field<'a, Vec<String>>, &'a str),
}

impl<'a> Filter<'a> {
    /// Match all children, short-circuiting at the first failure. An empty
    /// conjunction matches every document.
    pub fn and(filters: &'a [Self]) -> Self {
        Self {
            kind: FilterKind::And(filters),
        }
    }

    /// Match any child, short-circuiting at the first success. An empty
    /// disjunction matches no documents.
    pub fn or(filters: &'a [Self]) -> Self {
        Self {
            kind: FilterKind::Or(filters),
        }
    }

    /// Borrow this expression and negate its result, including for missing or
    /// differently typed values.
    pub fn negate(&'a self) -> Self {
        Self {
            kind: FilterKind::Not(self),
        }
    }

    pub fn matches(&self, metadata: &Metadata) -> bool {
        match self.kind {
            FilterKind::And(filters) => filters.iter().all(|filter| filter.matches(metadata)),
            FilterKind::Or(filters) => filters.iter().any(|filter| filter.matches(metadata)),
            FilterKind::Not(filter) => !filter.matches(metadata),
            FilterKind::Exists(name) => metadata.values.contains_key(name),
            FilterKind::StringEq(field, value) => {
                metadata.get(field).is_some_and(|stored| stored == value)
            }
            FilterKind::StringIn(field, values) => metadata
                .get(field)
                .is_some_and(|stored| values.contains(&stored.as_str())),
            FilterKind::BoolEq(field, value) => {
                metadata.get(field).is_some_and(|stored| *stored == value)
            }
            FilterKind::IntegerEq(field, value) => {
                metadata.get(field).is_some_and(|stored| *stored == value)
            }
            FilterKind::IntegerIn(field, values) => metadata
                .get(field)
                .is_some_and(|stored| values.contains(stored)),
            FilterKind::IntegerRange(field, bounds) => metadata
                .get(field)
                .is_some_and(|stored| bounds.contains(stored)),
            FilterKind::NumberEq(field, value) => {
                metadata.get(field).is_some_and(|stored| *stored == value)
            }
            FilterKind::NumberIn(field, values) => metadata
                .get(field)
                .is_some_and(|stored| values.contains(stored)),
            FilterKind::NumberRange(field, bounds) => metadata
                .get(field)
                .is_some_and(|stored| bounds.contains(stored)),
            FilterKind::StringsContains(field, value) => metadata
                .get(field)
                .is_some_and(|stored| stored.iter().any(|item| item == value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde::de::{IntoDeserializer, value::Error as ValueError};

    use super::*;

    const TITLE: Field<'static, String> = Field::new("title");
    const ACTIVE: Field<'static, bool> = Field::new("active");
    const COUNT: Field<'static, i64> = Field::new("count");
    const RATING: Field<'static, FiniteF64> = Field::new("rating");
    const TAGS: Field<'static, Vec<String>> = Field::new("tags");

    fn finite(value: f64) -> FiniteF64 {
        FiniteF64::new(value).unwrap()
    }

    fn example_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.insert(TITLE, "Rust guide");
        metadata.insert(ACTIVE, true);
        metadata.insert(COUNT, 5_i64);
        metadata.insert(RATING, finite(2.5));
        metadata.insert(TAGS, vec!["rust".to_owned(), "search".to_owned()]);
        metadata
    }

    #[test]
    fn typed_insert_and_get() {
        let empty = Metadata::default();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.get(TITLE), None);

        let metadata = example_metadata();
        assert!(!metadata.is_empty());
        assert_eq!(metadata.len(), 5);
        assert_eq!(metadata.get(TITLE).map(String::as_str), Some("Rust guide"));
        assert_eq!(metadata.get(ACTIVE), Some(&true));
        assert_eq!(metadata.get(COUNT), Some(&5));
        assert_eq!(metadata.get(RATING), Some(&finite(2.5)));
        assert_eq!(
            metadata.get(TAGS).unwrap(),
            &["rust".to_owned(), "search".to_owned()]
        );
        assert_eq!(metadata.get(Field::<bool>::new("title")), None);
        assert_eq!(metadata.get(Field::<String>::new("missing")), None);
        assert_eq!(metadata.clone(), metadata);
    }

    #[test]
    fn replacement_reuses_the_owned_name_and_can_change_type() {
        let mut metadata = Metadata::new();
        metadata.insert(TITLE, "old");
        let key_ptr = metadata.values.get_key_value("title").unwrap().0.as_ptr();
        metadata.insert(TITLE, "new");
        assert_eq!(metadata.get(TITLE).map(String::as_str), Some("new"));
        assert_eq!(
            metadata.values.get_key_value("title").unwrap().0.as_ptr(),
            key_ptr
        );

        let same_name = String::from("title");
        let boolean_title = Field::<bool>::new(&same_name);
        metadata.insert(boolean_title, false);
        assert_eq!(metadata.len(), 1);
        assert_eq!(metadata.get(TITLE), None);
        assert_eq!(metadata.get(boolean_title), Some(&false));
        assert_eq!(
            metadata.values.get_key_value("title").unwrap().0.as_ptr(),
            key_ptr
        );
    }

    #[test]
    fn field_names_and_values_are_owned_only_by_metadata() {
        struct NotCopyOrClone;
        fn assert_copy_clone<T: Copy + Clone>() {}
        assert_copy_clone::<Field<'_, NotCopyOrClone>>();

        let mut metadata = Metadata::new();
        {
            let name = String::from("dynamic");
            let value = String::from("borrowed input");
            let field = Field::<String>::new(&name);
            assert_eq!(field.name(), "dynamic");
            metadata.insert(field, value.as_str());
            assert_eq!(metadata.get(field), Some(&value));

            // A statically named field may still borrow local filter operands.
            let operand = String::from("borrowed input");
            let operands = [operand.as_str()];
            assert!(
                Field::<String>::new("dynamic")
                    .eq(&operand)
                    .matches(&metadata)
            );
            assert!(
                Field::<String>::new("dynamic")
                    .in_set(&operands)
                    .matches(&metadata)
            );
        }
        assert_eq!(
            metadata
                .get(Field::<String>::new("dynamic"))
                .map(String::as_str),
            Some("borrowed input")
        );
        assert!(
            Field::<NotCopyOrClone>::new("dynamic")
                .exists()
                .matches(&metadata)
        );
    }

    #[test]
    fn metadata_round_trips_through_json_and_postcard() {
        for metadata in [Metadata::default(), example_metadata()] {
            let json = serde_json::to_string(&metadata).unwrap();
            assert_eq!(serde_json::from_str::<Metadata>(&json).unwrap(), metadata);
            let bytes = postcard::to_stdvec(&metadata).unwrap();
            assert_eq!(postcard::from_bytes::<Metadata>(&bytes).unwrap(), metadata);
        }
    }

    #[test]
    fn finite_numbers_validate_and_round_trip() {
        for value in [
            0.0,
            -0.0,
            f64::MIN,
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            let number = finite(value);
            assert_eq!(number.get().to_bits(), value.to_bits());
            assert_eq!(f64::from(number).to_bits(), value.to_bits());
            assert_eq!(FiniteF64::try_from(value).unwrap(), number);
            let json = serde_json::to_string(&number).unwrap();
            assert_eq!(serde_json::from_str::<FiniteF64>(&json).unwrap(), number);
            let bytes = postcard::to_stdvec(&number).unwrap();
            assert_eq!(
                postcard::from_bytes::<FiniteF64>(&bytes)
                    .unwrap()
                    .get()
                    .to_bits(),
                value.to_bits()
            );
        }
    }

    #[test]
    fn deserialization_rejects_non_finite_numbers() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(FiniteF64::new(invalid), None);
            assert!(FiniteF64::try_from(invalid).is_err());
            let result: Result<FiniteF64, ValueError> =
                FiniteF64::deserialize(invalid.into_deserializer());
            assert!(result.is_err());

            let invalid_float = postcard::to_stdvec(&invalid).unwrap();
            assert!(postcard::from_bytes::<FiniteF64>(&invalid_float).is_err());

            // Corrupt the float payload in an otherwise valid metadata record.
            // This checks validation also applies through both containing types.
            let mut metadata = Metadata::new();
            metadata.insert(RATING, finite(0.0));
            let mut bytes = postcard::to_stdvec(&metadata).unwrap();
            let float_start = bytes.len() - invalid_float.len();
            bytes[float_start..].copy_from_slice(&invalid_float);
            assert!(postcard::from_bytes::<Metadata>(&bytes).is_err());
        }
    }

    #[test]
    fn string_equality_and_set_membership_are_exact() {
        let metadata = example_metadata();
        assert!(TITLE.eq("Rust guide").matches(&metadata));
        assert!(!TITLE.eq("rust guide").matches(&metadata));
        assert!(!TITLE.eq("Rust guide ").matches(&metadata));
        assert!(TITLE.in_set(&["other", "Rust guide"]).matches(&metadata));
        assert!(!TITLE.in_set(&["other", "rust guide"]).matches(&metadata));
        assert!(!TITLE.in_set(&[]).matches(&metadata));
    }

    #[test]
    fn boolean_equality() {
        let mut metadata = example_metadata();
        assert!(ACTIVE.eq(true).matches(&metadata));
        assert!(!ACTIVE.eq(false).matches(&metadata));
        metadata.insert(ACTIVE, false);
        assert!(ACTIVE.eq(false).matches(&metadata));
        assert!(!ACTIVE.eq(true).matches(&metadata));
    }

    #[test]
    fn integer_equality_set_membership_and_ranges() {
        use Bound::{Excluded, Included, Unbounded};

        let metadata = example_metadata();
        assert!(COUNT.eq(5).matches(&metadata));
        assert!(!COUNT.eq(4).matches(&metadata));
        assert!(COUNT.in_set(&[1, 5, 8]).matches(&metadata));
        assert!(!COUNT.in_set(&[1, 8]).matches(&metadata));
        assert!(!COUNT.in_set(&[]).matches(&metadata));
        assert!(COUNT.range(5..=5).matches(&metadata));
        assert!(!COUNT.range(5..5).matches(&metadata));
        assert!(COUNT.range(5..).matches(&metadata));
        assert!(COUNT.range(..=5).matches(&metadata));
        assert!(!COUNT.range(..5).matches(&metadata));

        for (lower, upper, expected) in [
            (Unbounded, Unbounded, true),
            (Included(5), Included(5), true),
            (Included(5), Excluded(5), false),
            (Excluded(5), Included(5), false),
            (Excluded(4), Excluded(6), true),
            (Excluded(5), Unbounded, false),
            (Unbounded, Excluded(5), false),
            (Included(6), Included(4), false),
        ] {
            assert_eq!(COUNT.range((lower, upper)).matches(&metadata), expected);
        }
    }

    #[test]
    fn integer_extreme_bounds_never_overflow() {
        use Bound::{Excluded, Unbounded};

        let mut metadata = Metadata::new();
        for value in [i64::MIN, i64::MIN + 1, 0, i64::MAX - 1, i64::MAX] {
            metadata.insert(COUNT, value);
            assert!(COUNT.range(..).matches(&metadata));
            assert!(COUNT.range(i64::MIN..=i64::MAX).matches(&metadata));
            assert!(COUNT.range(i64::MIN..).matches(&metadata));
            assert!(COUNT.range(..=i64::MAX).matches(&metadata));
            assert_eq!(
                COUNT.range(i64::MIN..=i64::MIN).matches(&metadata),
                value == i64::MIN
            );
            assert_eq!(
                COUNT.range(i64::MAX..=i64::MAX).matches(&metadata),
                value == i64::MAX
            );
            assert!(
                !COUNT
                    .range((Excluded(i64::MAX), Unbounded))
                    .matches(&metadata)
            );
            assert!(
                !COUNT
                    .range((Unbounded, Excluded(i64::MIN)))
                    .matches(&metadata)
            );
            assert_eq!(
                COUNT
                    .range((Excluded(i64::MIN), Excluded(i64::MAX)))
                    .matches(&metadata),
                value != i64::MIN && value != i64::MAX
            );
        }
    }

    #[test]
    fn number_equality_set_membership_and_ranges() {
        use Bound::{Excluded, Included, Unbounded};

        let mut metadata = example_metadata();
        let number = finite(2.5);
        assert!(RATING.eq(number).matches(&metadata));
        assert!(!RATING.eq(finite(3.0)).matches(&metadata));
        assert!(RATING.in_set(&[finite(0.0), number]).matches(&metadata));
        assert!(!RATING.in_set(&[finite(0.0)]).matches(&metadata));
        assert!(!RATING.in_set(&[]).matches(&metadata));
        assert!(RATING.range(number..=number).matches(&metadata));
        assert!(!RATING.range(number..number).matches(&metadata));
        assert!(RATING.range(number..).matches(&metadata));
        assert!(RATING.range(..=number).matches(&metadata));
        assert!(!RATING.range(..number).matches(&metadata));

        for (lower, upper, expected) in [
            (Unbounded, Unbounded, true),
            (Included(number), Included(number), true),
            (Included(number), Excluded(number), false),
            (Excluded(number), Included(number), false),
            (Excluded(finite(2.0)), Excluded(finite(3.0)), true),
            (Excluded(number), Unbounded, false),
            (Unbounded, Excluded(number), false),
            (Included(finite(3.0)), Included(finite(2.0)), false),
        ] {
            assert_eq!(RATING.range((lower, upper)).matches(&metadata), expected);
        }

        for value in [f64::MIN, -0.0, 0.0, f64::MAX] {
            metadata.insert(RATING, finite(value));
            assert!(
                RATING
                    .range(finite(f64::MIN)..=finite(f64::MAX))
                    .matches(&metadata)
            );
        }
        metadata.insert(RATING, finite(-0.0));
        assert!(RATING.eq(finite(0.0)).matches(&metadata));
        assert!(RATING.range(finite(0.0)..=finite(0.0)).matches(&metadata));
    }

    #[test]
    fn list_containment_is_exact() {
        let mut metadata = example_metadata();
        assert!(TAGS.contains("rust").matches(&metadata));
        assert!(TAGS.contains("search").matches(&metadata));
        assert!(!TAGS.contains("Rust").matches(&metadata));
        assert!(!TAGS.contains("rus").matches(&metadata));
        metadata.insert(TAGS, Vec::<String>::new());
        assert!(!TAGS.contains("").matches(&metadata));
        metadata.insert(
            TAGS,
            vec![String::new(), "rust".to_owned(), "rust".to_owned()],
        );
        assert!(TAGS.contains("").matches(&metadata));
        assert!(TAGS.contains("rust").matches(&metadata));
    }

    #[test]
    fn missing_and_wrong_types_fail_leaves_but_negation_matches() {
        let missing = Metadata::new();
        let mut wrong = Metadata::new();
        wrong.insert(Field::<bool>::new(TITLE.name()), true);
        wrong.insert(Field::<String>::new(ACTIVE.name()), "true");
        wrong.insert(Field::<FiniteF64>::new(COUNT.name()), finite(5.0));
        wrong.insert(Field::<i64>::new(RATING.name()), 2_i64);
        wrong.insert(Field::<String>::new(TAGS.name()), "rust");

        let numbers = [finite(2.0)];
        let filters = [
            TITLE.eq("Rust guide"),
            TITLE.in_set(&["Rust guide"]),
            ACTIVE.eq(true),
            COUNT.eq(5),
            COUNT.in_set(&[5]),
            COUNT.range(..),
            RATING.eq(finite(2.0)),
            RATING.in_set(&numbers),
            RATING.range(..),
            TAGS.contains("rust"),
        ];
        for filter in filters {
            assert!(!filter.matches(&missing), "{filter:?}");
            assert!(filter.negate().matches(&missing), "{filter:?}");
            assert!(!filter.matches(&wrong), "{filter:?}");
            assert!(filter.negate().matches(&wrong), "{filter:?}");
        }

        for exists in [
            TITLE.exists(),
            ACTIVE.exists(),
            COUNT.exists(),
            RATING.exists(),
            TAGS.exists(),
        ] {
            assert!(!exists.matches(&missing));
            assert!(exists.negate().matches(&missing));
            assert!(exists.matches(&wrong));
            assert!(!exists.negate().matches(&wrong));
        }
        assert!(!Field::<String>::new("TITLE").exists().matches(&wrong));
    }

    #[test]
    fn logical_operators_and_empty_expressions() {
        let metadata = example_metadata();
        assert!(Filter::and(&[]).matches(&metadata));
        assert!(!Filter::or(&[]).matches(&metadata));
        assert!(!Filter::and(&[]).negate().matches(&metadata));
        assert!(Filter::or(&[]).negate().matches(&metadata));
        assert!(Filter::and(&[ACTIVE.eq(true)]).matches(&metadata));
        assert!(!Filter::and(&[ACTIVE.eq(true), COUNT.eq(0)]).matches(&metadata));
        assert!(Filter::or(&[COUNT.eq(0), ACTIVE.eq(true)]).matches(&metadata));
        assert!(!Filter::or(&[COUNT.eq(0), ACTIVE.eq(false)]).matches(&metadata));
        let leaf = ACTIVE.eq(true);
        let negated = leaf.negate();
        assert!(negated.negate().matches(&metadata));
    }

    #[test]
    fn nested_borrowed_filters() {
        let mut metadata = example_metadata();
        let alternatives = [TITLE.eq("other"), TAGS.contains("rust")];
        let inactive = ACTIVE.eq(false);
        let clauses = [
            Filter::or(&alternatives),
            COUNT.range(1..=10),
            inactive.negate(),
        ];
        let filter = Filter::and(&clauses);
        assert!(filter.matches(&metadata));

        metadata.insert(ACTIVE, false);
        assert!(!filter.matches(&metadata));
        assert!(filter.negate().matches(&metadata));
        metadata.insert(ACTIVE, true);
        metadata.insert(COUNT, 11_i64);
        assert!(!filter.matches(&metadata));
        metadata.insert(COUNT, 5_i64);
        metadata.insert(TAGS, Vec::<String>::new());
        assert!(!filter.matches(&metadata));
        metadata.insert(TITLE, "other");
        assert!(filter.matches(&metadata));
    }
}
