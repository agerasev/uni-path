#[inline]
pub fn is_sep_char(b: char) -> bool {
    b == '/'
}

pub const MAIN_SEP_STR: &str = "/";
pub const MAIN_SEP: char = '/';
