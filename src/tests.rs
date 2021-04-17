use super::*;

use std::rc::Rc;
use std::sync::Arc;

macro_rules! t(
    ($path:expr, iter: $iter:expr) => (
        {
            let path = Path::new($path);

            // Forward iteration
            let comps = path.iter()
                .map(|p| p.to_string())
                .collect::<Vec<String>>();
            let exp: &[&str] = &$iter;
            let exps = exp.iter().map(|s| s.to_string()).collect::<Vec<String>>();
            assert!(comps == exps, "iter: Expected {:?}, found {:?}",
                    exps, comps);

            // Reverse iteration
            let comps = Path::new($path).iter().rev()
                .map(|p| p.to_string())
                .collect::<Vec<String>>();
            let exps = exps.into_iter().rev().collect::<Vec<String>>();
            assert!(comps == exps, "iter().rev(): Expected {:?}, found {:?}",
                    exps, comps);
        }
    );

    ($path:expr, has_root: $has_root:expr, is_absolute: $is_absolute:expr) => (
        {
            let path = Path::new($path);

            let act_root = path.has_root();
            assert!(act_root == $has_root, "has_root: Expected {:?}, found {:?}",
                    $has_root, act_root);

            let act_abs = path.is_absolute();
            assert!(act_abs == $is_absolute, "is_absolute: Expected {:?}, found {:?}",
                    $is_absolute, act_abs);
        }
    );

    ($path:expr, parent: $parent:expr, file_name: $file:expr) => (
        {
            let path = Path::new($path);

            let parent = path.parent().map(|p| p.to_str());
            let exp_parent: Option<&str> = $parent;
            assert!(parent == exp_parent, "parent: Expected {:?}, found {:?}",
                    exp_parent, parent);

            let file = path.file_name();
            let exp_file: Option<&str> = $file;
            assert!(file == exp_file, "file_name: Expected {:?}, found {:?}",
                    exp_file, file);
        }
    );

    ($path:expr, file_stem: $file_stem:expr, extension: $extension:expr) => (
        {
            let path = Path::new($path);

            let stem = path.file_stem();
            let exp_stem: Option<&str> = $file_stem;
            assert!(stem == exp_stem, "file_stem: Expected {:?}, found {:?}",
                    exp_stem, stem);

            let ext = path.extension();
            let exp_ext: Option<&str> = $extension;
            assert!(ext == exp_ext, "extension: Expected {:?}, found {:?}",
                    exp_ext, ext);
        }
    );

    ($path:expr, iter: $iter:expr,
                 has_root: $has_root:expr, is_absolute: $is_absolute:expr,
                 parent: $parent:expr, file_name: $file:expr,
                 file_stem: $file_stem:expr, extension: $extension:expr) => (
        {
            t!($path, iter: $iter);
            t!($path, has_root: $has_root, is_absolute: $is_absolute);
            t!($path, parent: $parent, file_name: $file);
            t!($path, file_stem: $file_stem, extension: $extension);
        }
    );
);

#[test]
fn into() {
    use std::borrow::Cow;

    let static_path = Path::new("/home/foo");
    let static_cow_path: Cow<'static, Path> = static_path.into();
    let pathbuf = PathBuf::from("/home/foo");

    {
        let path: &Path = &pathbuf;
        let borrowed_cow_path: Cow<'_, Path> = path.into();

        assert_eq!(static_cow_path, borrowed_cow_path);
    }

    let owned_cow_path: Cow<'static, Path> = pathbuf.into();

    assert_eq!(static_cow_path, owned_cow_path);
}

#[test]
pub fn test_decompositions_unix() {
    t!("",
    iter: [],
    has_root: false,
    is_absolute: false,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("foo",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("/",
    iter: ["/"],
    has_root: true,
    is_absolute: true,
    parent: None,
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("/foo",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo/",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("/foo/",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo/bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None
    );

    t!("/foo/bar",
    iter: ["/", "foo", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("/foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None
    );

    t!("///foo///",
    iter: ["/", "foo"],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("///foo///bar",
    iter: ["/", "foo", "bar"],
    has_root: true,
    is_absolute: true,
    parent: Some("///foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None
    );

    t!("./.",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("/..",
    iter: ["/", ".."],
    has_root: true,
    is_absolute: true,
    parent: Some("/"),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("../",
    iter: [".."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("foo/.",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo/..",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("foo/./",
    iter: ["foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some("foo"),
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo/./bar",
    iter: ["foo", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None
    );

    t!("foo/../",
    iter: ["foo", ".."],
    has_root: false,
    is_absolute: false,
    parent: Some("foo"),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("foo/../bar",
    iter: ["foo", "..", "bar"],
    has_root: false,
    is_absolute: false,
    parent: Some("foo/.."),
    file_name: Some("bar"),
    file_stem: Some("bar"),
    extension: None
    );

    t!("./a",
    iter: [".", "a"],
    has_root: false,
    is_absolute: false,
    parent: Some("."),
    file_name: Some("a"),
    file_stem: Some("a"),
    extension: None
    );

    t!(".",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("./",
    iter: ["."],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: None,
    file_stem: None,
    extension: None
    );

    t!("a/b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None
    );

    t!("a//b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None
    );

    t!("a/./b",
    iter: ["a", "b"],
    has_root: false,
    is_absolute: false,
    parent: Some("a"),
    file_name: Some("b"),
    file_stem: Some("b"),
    extension: None
    );

    t!("a/b/c",
    iter: ["a", "b", "c"],
    has_root: false,
    is_absolute: false,
    parent: Some("a/b"),
    file_name: Some("c"),
    file_stem: Some("c"),
    extension: None
    );

    t!(".foo",
    iter: [".foo"],
    has_root: false,
    is_absolute: false,
    parent: Some(""),
    file_name: Some(".foo"),
    file_stem: Some(".foo"),
    extension: None
    );
}

#[test]
pub fn test_stem_ext() {
    t!("foo",
    file_stem: Some("foo"),
    extension: None
    );

    t!("foo.",
    file_stem: Some("foo"),
    extension: Some("")
    );

    t!(".foo",
    file_stem: Some(".foo"),
    extension: None
    );

    t!("foo.txt",
    file_stem: Some("foo"),
    extension: Some("txt")
    );

    t!("foo.bar.txt",
    file_stem: Some("foo.bar"),
    extension: Some("txt")
    );

    t!("foo.bar.",
    file_stem: Some("foo.bar"),
    extension: Some("")
    );

    t!(".", file_stem: None, extension: None);

    t!("..", file_stem: None, extension: None);

    t!("", file_stem: None, extension: None);
}

#[test]
pub fn test_push() {
    macro_rules! tp(
        ($path:expr, $push:expr, $expected:expr) => ( {
            let mut actual = PathBuf::from($path);
            actual.push($push);
            assert!(actual.to_str() == $expected,
                    "pushing {:?} onto {:?}: Expected {:?}, got {:?}",
                    $push, $path, $expected, actual.to_str());
        });
    );

    tp!("", "foo", "foo");
    tp!("foo", "bar", "foo/bar");
    tp!("foo/", "bar", "foo/bar");
    tp!("foo//", "bar", "foo//bar");
    tp!("foo/.", "bar", "foo/./bar");
    tp!("foo./.", "bar", "foo././bar");
    tp!("foo", "", "foo/");
    tp!("foo", ".", "foo/.");
    tp!("foo", "..", "foo/..");
    tp!("foo", "/", "/");
    tp!("/foo/bar", "/", "/");
    tp!("/foo/bar", "/baz", "/baz");
    tp!("/foo/bar", "./baz", "/foo/bar/./baz");
}

#[test]
pub fn test_pop() {
    macro_rules! tp(
        ($path:expr, $expected:expr, $output:expr) => ( {
            let mut actual = PathBuf::from($path);
            let output = actual.pop();
            assert!(actual.to_str() == $expected && output == $output,
                    "popping from {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                    $path, $expected, $output,
                    actual.to_str(), output);
        });
    );

    tp!("", "", false);
    tp!("/", "/", false);
    tp!("foo", "", true);
    tp!(".", "", true);
    tp!("/foo", "/", true);
    tp!("/foo/bar", "/foo", true);
    tp!("foo/bar", "foo", true);
    tp!("foo/.", "", true);
    tp!("foo//bar", "foo", true);
}

#[test]
pub fn test_set_file_name() {
    macro_rules! tfn(
            ($path:expr, $file:expr, $expected:expr) => ( {
            let mut p = PathBuf::from($path);
            p.set_file_name($file);
            assert!(p.to_str() == $expected,
                    "setting file name of {:?} to {:?}: Expected {:?}, got {:?}",
                    $path, $file, $expected,
                    p.to_str());
        });
    );

    tfn!("foo", "foo", "foo");
    tfn!("foo", "bar", "bar");
    tfn!("foo", "", "");
    tfn!("", "foo", "foo");

    tfn!(".", "foo", "./foo");
    tfn!("foo/", "bar", "bar");
    tfn!("foo/.", "bar", "bar");
    tfn!("..", "foo", "../foo");
    tfn!("foo/..", "bar", "foo/../bar");
    tfn!("/", "foo", "/foo");
}

#[test]
pub fn test_set_extension() {
    macro_rules! tfe(
            ($path:expr, $ext:expr, $expected:expr, $output:expr) => ( {
            let mut p = PathBuf::from($path);
            let output = p.set_extension($ext);
            assert!(p.to_str() == $expected && output == $output,
                    "setting extension of {:?} to {:?}: Expected {:?}/{:?}, got {:?}/{:?}",
                    $path, $ext, $expected, $output,
                    p.to_str(), output);
        });
    );

    tfe!("foo", "txt", "foo.txt", true);
    tfe!("foo.bar", "txt", "foo.txt", true);
    tfe!("foo.bar.baz", "txt", "foo.bar.txt", true);
    tfe!(".test", "txt", ".test.txt", true);
    tfe!("foo.txt", "", "foo", true);
    tfe!("foo", "", "foo", true);
    tfe!("", "foo", "", false);
    tfe!(".", "foo", ".", false);
    tfe!("foo/", "bar", "foo.bar", true);
    tfe!("foo/.", "bar", "foo.bar", true);
    tfe!("..", "foo", "..", false);
    tfe!("foo/..", "bar", "foo/..", false);
    tfe!("/", "foo", "/", false);
}

#[test]
fn test_eq_receivers() {
    use std::borrow::Cow;

    let borrowed: &Path = Path::new("foo/bar");
    let mut owned: PathBuf = PathBuf::new();
    owned.push("foo");
    owned.push("bar");
    let borrowed_cow: Cow<'_, Path> = borrowed.into();
    let owned_cow: Cow<'_, Path> = owned.clone().into();

    macro_rules! t {
        ($($current:expr),+) => {
            $(
                assert_eq!($current, borrowed);
                assert_eq!($current, owned);
                assert_eq!($current, borrowed_cow);
                assert_eq!($current, owned_cow);
            )+
        }
    }

    t!(borrowed, owned, borrowed_cow, owned_cow);
}

#[test]
pub fn test_compare() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash<T: Hash>(t: T) -> u64 {
        let mut s = DefaultHasher::new();
        t.hash(&mut s);
        s.finish()
    }

    macro_rules! tc(
        ($path1:expr, $path2:expr, eq: $eq:expr,
         starts_with: $starts_with:expr, ends_with: $ends_with:expr,
         relative_from: $relative_from:expr) => ({
             let path1 = Path::new($path1);
             let path2 = Path::new($path2);

             let eq = path1 == path2;
             assert!(eq == $eq, "{:?} == {:?}, expected {:?}, got {:?}",
                     $path1, $path2, $eq, eq);
             assert!($eq == (hash(path1) == hash(path2)),
                     "{:?} == {:?}, expected {:?}, got {} and {}",
                     $path1, $path2, $eq, hash(path1), hash(path2));

             let starts_with = path1.starts_with(path2);
             assert!(starts_with == $starts_with,
                     "{:?}.starts_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                     $starts_with, starts_with);

             let ends_with = path1.ends_with(path2);
             assert!(ends_with == $ends_with,
                     "{:?}.ends_with({:?}), expected {:?}, got {:?}", $path1, $path2,
                     $ends_with, ends_with);

             let relative_from = path1.strip_prefix(path2)
                                      .map(|p| p.to_str())
                                      .ok();
             let exp: Option<&str> = $relative_from;
             assert!(relative_from == exp,
                     "{:?}.strip_prefix({:?}), expected {:?}, got {:?}",
                     $path1, $path2, exp, relative_from);
        });
    );

    tc!("", "",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo", "",
    eq: false,
    starts_with: true,
    ends_with: true,
    relative_from: Some("foo")
    );

    tc!("", "foo",
    eq: false,
    starts_with: false,
    ends_with: false,
    relative_from: None
    );

    tc!("foo", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/", "foo",
    eq: true,
    starts_with: true,
    ends_with: true,
    relative_from: Some("")
    );

    tc!("foo/bar", "foo",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("bar")
    );

    tc!("foo/bar/baz", "foo/bar",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("baz")
    );

    tc!("foo/bar", "foo/bar/baz",
    eq: false,
    starts_with: false,
    ends_with: false,
    relative_from: None
    );

    tc!("./foo/bar/", ".",
    eq: false,
    starts_with: true,
    ends_with: false,
    relative_from: Some("foo/bar")
    );
}

#[test]
fn test_components_debug() {
    let path = Path::new("/tmp");

    let mut components = path.components();

    let expected = "Components([RootDir, Normal(\"tmp\")])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([Normal(\"tmp\")])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);

    let _ = components.next().unwrap();
    let expected = "Components([])";
    let actual = format!("{:?}", components);
    assert_eq!(expected, actual);
}

#[cfg(unix)]
#[test]
fn test_iter_debug() {
    let path = Path::new("/tmp");

    let mut iter = path.iter();

    let expected = "Iter([\"/\", \"tmp\"])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([\"tmp\"])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);

    let _ = iter.next().unwrap();
    let expected = "Iter([])";
    let actual = format!("{:?}", iter);
    assert_eq!(expected, actual);
}

#[test]
fn into_boxed() {
    let orig: &str = "some/sort/of/path";
    let path = Path::new(orig);
    let boxed: Box<Path> = Box::from(path);
    let path_buf = path.to_owned().into_boxed_path().into_path_buf();
    assert_eq!(path, &*boxed);
    assert_eq!(&*boxed, &*path_buf);
    assert_eq!(&*path_buf, path);
}

/*
#[test]
fn display_format_flags() {
    assert_eq!(format!("a{:#<5}b", Path::new("").display()), "a#####b");
    assert_eq!(format!("a{:#<5}b", Path::new("a").display()), "aa####b");
}
*/

#[test]
fn into_rc() {
    let orig = "hello/world";
    let path = Path::new(orig);
    let rc: Rc<Path> = Rc::from(path);
    let arc: Arc<Path> = Arc::from(path);

    assert_eq!(&*rc, path);
    assert_eq!(&*arc, path);

    let rc2: Rc<Path> = Rc::from(path.to_owned());
    let arc2: Arc<Path> = Arc::from(path.to_owned());

    assert_eq!(&*rc2, path);
    assert_eq!(&*arc2, path);
}
