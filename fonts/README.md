# Fonts

The figures are set in Nimbus Sans, URW++'s Helvetica clone from the urw-base35 collection (Copyright 2014 by (URW)++ Design & Development). Licence: GNU Affero General Public License v3 with URW's exemption for including the fonts in PostScript or PDF documents (`LICENSE`, `COPYING`, both copied verbatim from URW's repository). This applies to every file in this folder, the originals and the converted ones alike. The code of this repository is under its own BSD 3-Clause licence (root `LICENSE`), which does not cover this folder; the code runs without these fonts (matplotlib falls back to Arial, Helvetica or DejaVu Sans).

## Originals

`original/NimbusSans-{Regular,Bold,Italic,BoldItalic}.otf`: unmodified, from https://github.com/ArtifexSoftware/urw-base35-fonts, `fonts/` directory, commit c1e3ec2886de92b8898f90e4dad9926549bb28a5 (2017-08-17), downloaded 2026-10-13.

## Modified files

`NimbusSansTT-{Regular,Bold,Italic,BoldItalic}.ttf` were made from the originals on 2026-10-07 by Felix Schaumann with fontTools: the cubic (CFF) outlines were converted to quadratic TrueType outlines, the family name was changed to "Nimbus Sans TT" (PostScript names `NimbusSansTT-*`), everything else was left as it was. Checked against the originals on 2026-10-13: identical glyph set, character map, units per em and advance widths (one Cyrillic glyph, `afii10084`, has a different advance in the two italic faces). The TrueType form is what matplotlib embeds into PDFs as editable text (`pdf.fonttype 42`), which the journal requires. These modified files are distributed under the same licence and exemption as the originals. Not affiliated with or endorsed by URW++ or Artifex.

`src/teu_functions.py` registers the TTFs with matplotlib on import; nothing needs to be installed.
