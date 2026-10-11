use std::collections::HashSet;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub name: String,
    pub documents: Vec<Document>,
    pub queries: Vec<Query>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Document {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub relevant_document_ids: Vec<String>,
}

impl Dataset {
    pub fn parse(json: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let dataset: Self = serde_json::from_str(json)?;
        dataset.validate()?;
        Ok(dataset)
    }

    fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() || self.documents.is_empty() || self.queries.is_empty() {
            return Err("dataset needs a name, documents, and queries".into());
        }
        let mut document_ids = HashSet::new();
        for document in &self.documents {
            if document.id.trim().is_empty() || document.text.trim().is_empty() {
                return Err("documents need nonblank IDs and text".into());
            }
            if !document_ids.insert(document.id.as_str()) {
                return Err(format!("duplicate document ID: {}", document.id));
            }
        }
        let mut query_ids = HashSet::new();
        for query in &self.queries {
            if query.id.trim().is_empty() || query.text.trim().is_empty() {
                return Err("queries need nonblank IDs and text".into());
            }
            if !query_ids.insert(query.id.as_str()) {
                return Err(format!("duplicate query ID: {}", query.id));
            }
            if query.relevant_document_ids.is_empty() {
                return Err(format!("query {} has no relevance labels", query.id));
            }
            let mut relevant = HashSet::new();
            for doc_id in &query.relevant_document_ids {
                if !document_ids.contains(doc_id.as_str()) {
                    return Err(format!(
                        "query {} references unknown document: {doc_id}",
                        query.id
                    ));
                }
                if !relevant.insert(doc_id) {
                    return Err(format!(
                        "query {} repeats relevant document: {doc_id}",
                        query.id
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Dataset {
        Dataset::parse(r#"{
            "name": "test",
            "documents": [{"id": "answer", "text": "An answer"}],
            "queries": [{"id": "question", "text": "A question", "relevant_document_ids": ["answer"]}]
        }"#).unwrap()
    }

    #[test]
    fn bundled_dataset_has_valid_labels_even_if_documents_are_reordered() {
        let mut dataset = Dataset::parse(include_str!("../../eval/dataset.json")).unwrap();
        dataset.documents.reverse();
        dataset.validate().unwrap();
        assert!(
            dataset
                .queries
                .iter()
                .any(|query| query.relevant_document_ids.len() > 1)
        );
    }

    #[test]
    fn rejects_unknown_duplicate_and_missing_relevance_labels() {
        for labels in [vec!["missing"], vec!["answer", "answer"], vec![]] {
            let mut dataset = valid();
            dataset.queries[0].relevant_document_ids =
                labels.into_iter().map(String::from).collect();
            assert!(dataset.validate().is_err());
        }
    }

    #[test]
    fn rejects_duplicate_ids_and_blank_text() {
        let mut dataset = valid();
        dataset.documents.push(Document {
            id: "answer".into(),
            text: "another".into(),
        });
        assert!(
            dataset
                .validate()
                .unwrap_err()
                .contains("duplicate document ID")
        );
        let mut dataset = valid();
        dataset.queries.push(Query {
            id: "question".into(),
            text: "another".into(),
            relevant_document_ids: vec!["answer".into()],
        });
        assert!(
            dataset
                .validate()
                .unwrap_err()
                .contains("duplicate query ID")
        );
        let mut dataset = valid();
        dataset.documents[0].text = " \n".into();
        assert!(dataset.validate().is_err());
        let mut dataset = valid();
        dataset.queries[0].text = " \t".into();
        assert!(dataset.validate().is_err());
    }

    #[test]
    fn rejects_empty_datasets_and_misspelled_fields() {
        let mut dataset = valid();
        dataset.documents.clear();
        assert!(dataset.validate().is_err());
        let mut dataset = valid();
        dataset.queries.clear();
        assert!(dataset.validate().is_err());
        assert!(Dataset::parse(r#"{"name": "test", "documents": [], "querys": []}"#).is_err());
    }
}
