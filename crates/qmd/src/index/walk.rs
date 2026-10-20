//! File discovery for indexing: glob pattern + ignore patterns, hard-coded
//! directory excludes, dotfile filtering, no symlink following. Mirrors the
//! TS fast-glob configuration in `reindexCollection`.

use std::path::Path;

use globset::{GlobBuilder, GlobSet, GlobSetBuilder};

use crate::error::{Error, Result};

/// Directories always excluded from indexing.
const EXCLUDE_DIRS: [&str; 6] = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

fn build_globset(patterns: &[String]) -> Result<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        let glob = GlobBuilder::new(pattern)
            .literal_separator(true)
            .build()
            .map_err(|e| Error::Config(format!("invalid glob pattern '{pattern}': {e}")))?;
        builder.add(glob);
    }
    builder
        .build()
        .map_err(|e| Error::Config(format!("invalid glob set: {e}")))
}

/// Files found by [`walk_files`].
#[derive(Debug, Default)]
pub struct Walk {
    /// Relative paths (forward slashes), sorted.
    pub files: Vec<String>,
    /// Entries the walk could not read. When non-zero, `files` may miss files
    /// that still exist, so absence from it does not mean deleted.
    pub errors: usize,
}

/// Walk `root` and return relative paths (forward slashes) matching `pattern`
/// and not matching any ignore pattern. Hidden files/directories and the
/// hard-coded exclude dirs are always skipped; symlinks are not followed.
///
/// # Errors
/// Fails on an invalid glob, or when `root` is not a readable directory.
pub fn walk_files(root: &Path, pattern: &str, ignore_patterns: &[String]) -> Result<Walk> {
    if !std::fs::metadata(root)?.is_dir() {
        return Err(Error::Config(format!(
            "collection path is not a directory: {}",
            root.display()
        )));
    }
    let include = build_globset(std::slice::from_ref(&pattern.to_owned()))?;
    let ignore_globs: Vec<String> = EXCLUDE_DIRS
        .iter()
        .map(|d| format!("**/{d}/**"))
        .chain(ignore_patterns.iter().cloned())
        .collect();
    let ignore = build_globset(&ignore_globs)?;

    let walker = ignore::WalkBuilder::new(root)
        .hidden(true)
        .ignore(false)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .parents(false)
        .follow_links(false)
        .build();

    let mut walk = Walk::default();
    for entry in walker {
        let Ok(entry) = entry else {
            walk.errors += 1;
            continue;
        };
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let Ok(relative) = entry.path().strip_prefix(root) else {
            continue;
        };
        let rel = relative.to_string_lossy().replace('\\', "/");
        if include.is_match(&rel) && !ignore.is_match(&rel) {
            walk.files.push(rel);
        }
    }
    walk.files.sort();
    Ok(walk)
}

/// [`walk_files`] without the error count.
///
/// # Errors
/// As [`walk_files`].
pub fn list_files(root: &Path, pattern: &str, ignore_patterns: &[String]) -> Result<Vec<String>> {
    Ok(walk_files(root, pattern, ignore_patterns)?.files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn touch(path: &Path) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, "# x\n").unwrap();
    }

    #[test]
    fn walk_respects_pattern_excludes_and_dotfiles() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        touch(&root.join("a.md"));
        touch(&root.join("sub/b.md"));
        touch(&root.join("sub/c.txt"));
        touch(&root.join("node_modules/dep/readme.md"));
        touch(&root.join(".hidden/d.md"));
        touch(&root.join("sub/.dotfile.md"));
        touch(&root.join("ignored/e.md"));

        let files = list_files(root, "**/*.md", &["ignored/**".to_owned()]).unwrap();
        assert_eq!(files, vec!["a.md".to_owned(), "sub/b.md".to_owned()]);
    }

    #[test]
    fn star_does_not_cross_directories() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        touch(&root.join("a.md"));
        touch(&root.join("sub/b.md"));
        touch(&root.join("keep/c.md"));
        touch(&root.join("keep/deep/d.md"));

        assert_eq!(list_files(root, "*.md", &[]).unwrap(), vec!["a.md"]);
        let files = list_files(root, "**/*.md", &["keep/*".to_owned()]).unwrap();
        assert_eq!(files, vec!["a.md", "keep/deep/d.md", "sub/b.md"]);
    }

    #[test]
    fn missing_or_non_directory_root_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(list_files(&dir.path().join("gone"), "**/*.md", &[]).is_err());
        touch(&dir.path().join("file.md"));
        assert!(list_files(&dir.path().join("file.md"), "**/*.md", &[]).is_err());
    }
}
