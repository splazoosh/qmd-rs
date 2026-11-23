//! Live pull of the default embedding model through each download route.
//! Opt-in twice: run with `--ignored` and name a throwaway cache directory in
//! `QMD_LIVE_HF_CACHE`. Needs the network and an `hf` CLI on `PATH`; sends the
//! Hugging Face token from the environment when one is set.

use std::path::Path;
use std::time::Instant;

use qmd::models::hf::{HfSetup, Route};
use qmd::models::{DEFAULT_EMBED_MODEL, inspect_gguf_file, model_path, pull_model_with};

#[test]
#[ignore = "downloads ~300 MB from Hugging Face; set QMD_LIVE_HF_CACHE"]
fn both_routes_land_the_default_model_at_the_same_relative_path() {
    let root = std::env::var_os("QMD_LIVE_HF_CACHE").expect("QMD_LIVE_HF_CACHE names a cache");
    let root = Path::new(&root);
    let setup = HfSetup::from_env();
    assert!(setup.cli.is_some(), "no hf CLI on PATH");
    let builtin = HfSetup {
        cli: None,
        ..setup.clone()
    };

    for (name, setup) in [("cli", &setup), ("builtin", &builtin)] {
        let cache = root.join(name);
        let started = Instant::now();
        let result = pull_model_with(setup, DEFAULT_EMBED_MODEL, &cache, false).unwrap();
        println!(
            "{name}: {} in {:.1}s, {} bytes, token {}",
            result
                .route
                .as_ref()
                .map_or("cached".to_owned(), Route::to_string),
            started.elapsed().as_secs_f64(),
            result.size_bytes,
            setup.token_status(),
        );
        assert_eq!(result.cli_failure, None);
        assert_eq!(result.route, Some(setup.route()));
        assert_eq!(result.path, model_path(DEFAULT_EMBED_MODEL, &cache));
        assert!(inspect_gguf_file(&result.path).is_valid());
    }
}
