### Docs: PyPI's description writes its links in a plain-text-safe form

PyPI also serves the description as raw text (its JSON API), and some tools read it that way. Five links in `PYPI.md` ended
in ")" or "." straight after the URL, so a plain-text autolinker took the punctuation into the URL and requested paths
that 404. Those five now read `text: <url>`. Angle brackets end the URL, and markdown still renders them as links. The
rendered page is unchanged; the fix reaches PyPI with the next release.
