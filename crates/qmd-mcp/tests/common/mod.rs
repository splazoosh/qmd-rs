//! The fixture index both test binaries serve.

use std::path::Path;

use qmd::config::{Collection, CollectionConfig, ContextMap};
use qmd::embed::{EmbedOptions, EmbeddingBackend, MockEmbedder, generate_embeddings};
use qmd::index::reindex_collection;
use qmd::store::Store;
use qmd_mcp::Worker;

/// A temp index holding three documents in two collections.
pub struct Fixture {
    _dir: tempfile::TempDir,
    pub config_path: std::path::PathBuf,
    pub db_path: std::path::PathBuf,
}

fn write(path: &Path, body: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, body).unwrap();
}

impl Fixture {
    pub fn build() -> Self {
        Self::build_with(None)
    }

    /// The fixture with `recipes` set to section documents at `level`, and
    /// a two-section `recipes/soup.md`.
    pub fn build_with(recipe_sections: Option<u8>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        let notes = root.join("notes");
        write(
            &notes.join("raft.md"),
            "# Raft Consensus\n\nRaft elects a leader; followers replicate the log.\n\
             An election starts when a follower stops hearing heartbeats.\n",
        );
        write(
            &notes.join("sub/bread.md"),
            "# Sourdough Basics\n\nFeed the starter, fold the dough, bake at high heat.\n",
        );
        let recipes = root.join("recipes");
        write(
            &recipes.join("stock.md"),
            "# Chicken Stock\n\nSimmer bones for six hours; skim the surface.\n",
        );
        if recipe_sections.is_some() {
            write(
                &recipes.join("soup.md"),
                "# Soup\n\n## Broth\n\nStrain the broth.\n\n## Garnish\n\nChop the parsley finely.\n",
            );
        }

        let mut context = ContextMap::new();
        context.insert("sub".to_owned(), "baking notes".to_owned());
        let mut config = CollectionConfig {
            global_context: Some("A test corpus.".to_owned()),
            ..CollectionConfig::default()
        };
        config.collections.insert(
            "notes".to_owned(),
            Collection {
                path: notes.to_string_lossy().into_owned(),
                pattern: "**/*.md".to_owned(),
                ignore: None,
                context: Some(context),
                update: None,
                include_by_default: Some(true),
                chunk_tokens: None,
                chunk_boundary: None,
                section_level: None,
            },
        );
        config.collections.insert(
            "recipes".to_owned(),
            Collection {
                path: recipes.to_string_lossy().into_owned(),
                pattern: "**/*.md".to_owned(),
                ignore: None,
                context: None,
                update: None,
                include_by_default: Some(true),
                chunk_tokens: None,
                chunk_boundary: None,
                section_level: recipe_sections,
            },
        );

        let config_path = root.join("index.yml");
        qmd::config::save_config(&config_path, &config).unwrap();
        let db_path = root.join("index.sqlite");

        let store = Store::open(&db_path).unwrap();
        store.sync_config_to_db(&config).unwrap();
        reindex_collection(&store, &notes, "**/*.md", "notes", &[], None).unwrap();
        reindex_collection(&store, &recipes, "**/*.md", "recipes", &[], None).unwrap();
        let backend = MockEmbedder::default();
        generate_embeddings(&store, &backend, &EmbedOptions::default()).unwrap();

        Self {
            _dir: dir,
            config_path,
            db_path,
        }
    }

    pub fn worker(&self) -> Worker {
        let factory =
            Box::new(|_: &str| Ok(Box::new(MockEmbedder::default()) as Box<dyn EmbeddingBackend>));
        Worker::spawn(
            &self.config_path,
            &self.db_path,
            qmd::embed::Embedder::model_uri(&MockEmbedder::default()).to_owned(),
            qmd_mcp::Factories {
                embedder: factory,
                reranker: Box::new(|_: &str| {
                    Ok(Box::new(qmd::rerank::MockReranker) as Box<dyn qmd::rerank::Reranker>)
                }),
            },
        )
        .unwrap()
    }
}
