use kestrel::{IvfBuildOptions, IvfIndex, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut exact = VectorIndex::new(2)?;
    for (id, vector) in [
        (10, [1.0, 0.0]),
        (20, [1.0, 0.1]),
        (30, [-1.0, 0.0]),
        (40, [-1.0, 0.1]),
    ] {
        exact.insert(id, &vector)?;
    }
    let ivf = IvfIndex::build(
        &exact,
        IvfBuildOptions {
            clusters: 2,
            ..IvfBuildOptions::default()
        },
    )?;
    println!(
        "Cluster sizes: {:?}",
        ivf.cluster_sizes().collect::<Vec<_>>()
    );
    for probes in [1, 2] {
        println!(
            "{probes} probe(s): {:?}",
            ivf.search(&[1.0, 0.0], 4, probes)?
        );
    }
    Ok(())
}
