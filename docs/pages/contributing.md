# Contributing tutorials

## Preview the documentation

From the repository root, install the documentation dependencies in your Python
environment and start the preview server:

```bash
python -m pip install -r docs/requirements.txt
python -m mkdocs serve -f docs/mkdocs.yml
```

Open the local URL printed by MkDocs. A documentation-only build does not require
Finitewave, PyVista, or the simulation dependencies: notebooks are converted from
their saved content without execution.

## Add or update a notebook

1. Put the notebook in `Tutorials/` and its companion inputs in `Tutorials/data/`.
2. Start with a short introduction, prerequisites, and the expected result.
3. Use one main heading and section headings to organize the steps.
4. Run the notebook in its simulation environment, check the outputs, and save it.
5. Add its filename to `nav` in `docs/mkdocs.yml` and link it from the start page.
6. Build the site and inspect the resulting page:

```bash
python -m mkdocs build --strict -f docs/mkdocs.yml
```

The generated website goes into `docs/site/`. Do not commit this directory. The CI
workflow builds the documentation and uploads the site as an artifact.

## Figures, videos, and data

Saved notebook outputs appear on the website. Use static screenshots for PyVista
figures; interactive VTK windows require a running Python environment. Embedded
videos are portable but increase notebook size. Companion files under `data/`
and `output/` are copied into the site, so keep published assets reasonably small
and check that links work in the rendered page.

MkDocs Markdown extensions apply to `.md` pages. Notebook Markdown is rendered
by nbconvert, so use standard notebook Markdown there.

## Publish to GitHub Pages

The configuration assumes `https://finitewave.github.io/Finitewave-Tools/`.
Change `site_url` and `repo_url` if hosting elsewhere.

After reviewing a successful build, maintainers can publish manually:

```bash
python -m mkdocs gh-deploy -f docs/mkdocs.yml
```

Configure the repository's Pages settings to serve the root of the `gh-pages`
branch. The build workflow does not deploy automatically.

## Configuration references

- [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/)
- [MkDocs themes](https://www.mkdocs.org/user-guide/choosing-your-theme/)
- [Notebook plugin](https://github.com/danielfrg/mkdocs-jupyter)

## Site appearance

The site uses Material for MkDocs with a teal-to-blue header, purple accents, expanded tutorial
navigation, and a separate page contents sidebar. Theme settings are in
`docs/mkdocs.yml` and `docs/pages/stylesheets/colorful.css`; the notebook download button is in `docs/overrides/main.html`.
Install the theme with the other dependencies in `docs/requirements.txt`.

## Documentation layout

All documentation configuration, Markdown pages, theme overrides, and styles
live in `docs/`. The notebooks and their input/output assets remain in
`Tutorials/` so their local execution paths stay valid. The hook in
`docs/hooks/tutorials.py` includes those files directly during a build; it does
not create notebook copies in the source tree. Both directories are watched
by the preview server.

Code blocks allow approximately 100 monospace characters on wide screens.
Longer lines scroll horizontally, including on smaller screens. This changes
the display width, not Python formatting or notebook source lines.

The GitHub Actions workflow stays in `.github/workflows/`, where GitHub requires
it, and reads the configuration and dependencies from `docs/`.
