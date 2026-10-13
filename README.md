# personal-blog

Source of [blog.tbeolet.com](https://blog.tbeolet.com/), my technical blog about AI, robotics and
reinforcement learning. Built with [Hugo](https://gohugo.io/) (extended) and the
[PaperMod](https://github.com/adityatelange/hugo-PaperMod) theme, deployed to GitHub Pages.

## Local development

Clone with the theme submodule, then start the dev server (drafts included):

```bash
git clone --recurse-submodules https://github.com/TristanBEOLET/personal-blog.git
cd personal-blog
hugo server -D
```

If you already cloned without submodules: `git submodule update --init --recursive`.

The site is then served at http://localhost:1313/.

## Writing a post

```bash
hugo new content posts/<slug>.md
```

Posts are published at `/posts/<slug>/`. Front matter:

```yaml
---
title: "Post title"
date: 2026-01-01
tags: ["rl", "paper"]
draft: false
description: "One-line summary, used for SEO."
---
```

- A post is either a single Markdown file or a page bundle folder (`posts/<slug>/index.md`) with its images.
- Math uses LaTeX: `$...$` or `\(...\)` inline, `$$...$$` or `\[...\]` for display (rendered by MathJax).
- Put `<!--more-->` after the intro to control the summary shown in the post list.
- Code for project posts lives in separate repositories, linked from the post.
- Videos: put the `.mp4` in the page bundle and use `{{< video src="clip.mp4" poster="clip.jpg" caption="..." >}}`.

### Series

Posts that belong together share a `series` entry in their front matter:

```yaml
series: ["SO-100 tower stacking"]
linkTitle: "Does a reverse curriculum help?"   # short title used in the series box and page
```

- Each post of a series gets a box at the top listing every part, in date order.
- `/series/` lists all series; `/series/<name>/` lists one series' posts, oldest first.
- To give a series a title and description, add `content/series/<urlized-name>/_index.md`
  (e.g. `content/series/so-100-tower-stacking/_index.md`).

## Layout

| Path | Purpose |
|---|---|
| `hugo.toml` | Site configuration |
| `content/` | Posts and pages |
| `layouts/_partials/` | Overrides of PaperMod partials (fonts, MathJax, tags in post list) and the series box (`series_nav.html`) |
| `layouts/single.html` | Copy of PaperMod's `single.html` with one added line: the series box. Re-check it after a theme update |
| `layouts/series/` | `/series/` and series pages (posts in reading order) |
| `layouts/_shortcodes/` | `video` shortcode |
| `assets/css/extended/custom.css` | Custom styles (colors, fonts, post list) |
| `static/` | Files copied as-is (favicon) |
| `themes/PaperMod/` | Theme submodule — never edit; override by mirroring paths instead |

## Deployment

Pushing to `main` runs `.github/workflows/hugo.yml`, which builds the site and deploys it to
GitHub Pages. Pull requests run the same build as a check, without deploying.

To update the theme: `git submodule update --remote themes/PaperMod`, then check the site
locally before committing.
