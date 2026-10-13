# CLAUDE.md — Technical blog (Hugo + PaperMod)

This repo is a personal technical blog about AI, robotics and reinforcement learning.
All design and tooling decisions below are already made. Your job is to build the site
exactly as described, verify it builds, and commit it. Do not re-open these decisions
unless something is technically impossible; if so, explain why and propose the closest option.

## Site identity

- Owner: Tristan Beolet
- Site title: `Tristan Beolet`
- Blog URL: `https://blog.tbeolet.com/` (root domain `tbeolet.com` is kept free for later use)
- GitHub: https://github.com/TristanBEOLET
- LinkedIn: https://www.linkedin.com/in/tristan-beolet

## Decisions (do not change)

- Static site generator: **Hugo extended** (latest stable release).
- Theme: **PaperMod**, added as a git submodule in `themes/PaperMod`. Never edit files
  inside `themes/`; override by mirroring paths in the repo's own `layouts/` and `assets/`.
- Hosting: **GitHub Pages**, deployed by a GitHub Actions workflow on push to `main`.
- Domain: custom domain `blog.tbeolet.com` on Cloudflare (DNS handled manually by the owner).
- Math: **LaTeX via MathJax 3**, with Goldmark passthrough so Markdown doesn't mangle equations.
- Style: warm "paper" look, forest-green accents, serif headings, **no gradients**,
  posts listed as a simple divided list (not cards), tags shown as small tinted boxes.
- Colors: light accent `#1F7D53`, dark accent `#4E9F3D`. No other accent colors.
- Fonts: headings in Source Serif 4, code in JetBrains Mono (Google Fonts), body in system UI font.
- No notebooks on the site; code lives in separate GitHub repos linked from posts.

## Setup steps

### 1. Scaffold

```bash
hugo new site . --force --format toml
git submodule add --depth=1 https://github.com/adityatelange/hugo-PaperMod.git themes/PaperMod
```

Create `.gitignore`:

```
public/
resources/_gen/
.hugo_build.lock
```

### 2. `hugo.toml`

```toml
baseURL = "https://blog.tbeolet.com/"
languageCode = "en-us"
title = "Tristan Beolet"
theme = "PaperMod"

[params]
  defaultTheme = "auto"
  ShowReadingTime = true
  ShowToc = true
  ShowCodeCopyButtons = true
  ShowPostNavLinks = true
  math = true

  [params.assets]
    favicon = "/favicon.svg"

  [params.homeInfoParams]
    Title = "Notes on robots and learning agents"
    Content = "Write-ups of things I build, break and fix."

  [[params.socialIcons]]
    name = "github"
    url = "https://github.com/TristanBEOLET"
  [[params.socialIcons]]
    name = "linkedin"
    url = "https://www.linkedin.com/in/tristan-beolet"

[menu]
  [[menu.main]]
    name = "Posts"
    url = "/posts/"
    weight = 1
  [[menu.main]]
    name = "About"
    url = "/about/"
    weight = 2

[markup.goldmark.extensions.passthrough]
  enable = true
  [markup.goldmark.extensions.passthrough.delimiters]
    block = [['\[', '\]'], ['$$', '$$']]
    inline = [['\(', '\)'], ['$', '$']]
```

### 3. Head partial (fonts + MathJax)

Check where PaperMod keeps its partials (`themes/PaperMod/layouts/partials/` or
`themes/PaperMod/layouts/_partials/`, depending on version) and mirror that folder.
Create `extend_head.html` there:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

{{ if or .Params.math .Site.Params.math }}
<script>
  MathJax = {
    tex: {
      displayMath: [['\\[', '\\]'], ['$$', '$$']],
      inlineMath: [['\\(', '\\)'], ['$', '$']]
    }
  };
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
{{ end }}
```

### 4. Tags in the post list

PaperMod doesn't show tags in the post list by default. Find the theme's `post_meta.html`
(`find themes/PaperMod/layouts -name post_meta.html`), copy it to the same relative path
under the repo's `layouts/`, and append at the very end of the copy:

```html
{{- with .Params.tags }}
<span class="meta-tags">
  {{- range . }}<span class="meta-tag">{{ . }}</span>{{ end }}
</span>
{{- end }}
```

Tags must be plain `<span>`s, not links: PaperMod wraps each list entry in an overlay link.

### 5. Custom CSS

Create `assets/css/extended/custom.css` (PaperMod bundles it automatically):

```css
:root {
  --theme: #fbfaf6;
  --entry: #fbfaf6;
  --primary: #1d2420;
  --secondary: #5d665f;
  --tertiary: #e4e2da;
  --content: #262d28;
  --code-block-bg: #1b221e;
  --code-bg: #eeefe7;
  --border: #e4e2da;
  --accent: #1F7D53;
}

.dark, :root[data-theme="dark"] {
  --theme: #141815;
  --entry: #141815;
  --primary: #e6e8e2;
  --secondary: #a3aba2;
  --tertiary: #2a312b;
  --content: #d6dad3;
  --code-block-bg: #0d100e;
  --code-bg: #1e241f;
  --border: #2a312b;
  --accent: #4E9F3D;
}

body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
code, pre { font-family: "JetBrains Mono", ui-monospace, monospace; }

h1, h2, h3, .post-title, .entry-header h2, .logo a {
  font-family: "Source Serif 4", Georgia, serif;
  font-weight: 600;
}

.post-entry {
  background: transparent;
  border: none;
  border-top: 1px solid var(--border);
  border-radius: 0;
  margin-bottom: 0;
  padding: 32px 0;
}
.post-entry:active { transform: none; }
.post-entry:hover .entry-header h2 { color: var(--accent); }

.post-content a {
  color: var(--accent);
  box-shadow: none;
  text-decoration: underline;
  text-decoration-thickness: 1px;
  text-underline-offset: 3px;
}
.post-content a:hover { text-decoration-thickness: 2px; }

#menu .active { color: var(--accent); border-bottom-color: var(--accent); }

.meta-tags { margin-left: 8px; }

.meta-tag,
.post-tags a {
  display: inline-block;
  font-size: 12px;
  line-height: 1.6;
  padding: 1px 8px;
  margin: 2px 4px 2px 0;
  border: none;
  border-radius: 6px;
  color: var(--accent);
  background: color-mix(in srgb, var(--accent) 12%, transparent);
}

.post-tags a:hover {
  background: color-mix(in srgb, var(--accent) 22%, transparent);
}

::selection { background: var(--accent); color: #fff; }
```

If a selector doesn't match the installed PaperMod version, inspect the theme's templates
and adjust the selector — keep the visual intent identical.

### 6. Favicon

Create `static/favicon.svg`: a simple flat rounded square in `#1F7D53` with the initials
"TB" in white. No gradients.

### 7. Starter content

- `content/about.md`: short placeholder About page (who, what I work on, links), `draft: false`.
- `content/posts/hello-world.md` with `draft: false`, `tags: ["meta"]`, and a body that
  tests the setup: one inline equation like `$\pi_\theta(a|s)$`, one display equation like
  `$$ J(\theta) = \mathbb{E}\left[\sum_t \gamma^t r_t\right] $$`, a Python code block,
  and a link.

### 8. Deployment workflow

Create `.github/workflows/hugo.yml` based on GitHub's official Hugo Pages workflow:
trigger on push to `main` and `workflow_dispatch`; permissions `contents: read`,
`pages: write`, `id-token: write`; install **Hugo extended** (pin `HUGO_VERSION` to the
latest stable release); checkout with `submodules: recursive` and `fetch-depth: 0`;
`actions/configure-pages`; build with
`hugo --gc --minify --baseURL "${{ steps.pages.outputs.base_url }}/"`;
`actions/upload-pages-artifact` with path `./public`; then a `deploy` job using
`actions/deploy-pages`. Use the current major versions of these actions.

### 9. Verify and commit

- Run `hugo --gc --minify` and fix any errors or warnings.
- Check the generated HTML: tags appear in the post list, MathJax script is present,
  custom CSS is in the bundled stylesheet.
- Commit with a clear message and push to `main`.

## Manual steps for the owner (cannot be done from a session)

1. GitHub repo → Settings → Pages → Source: **GitHub Actions**.
2. Cloudflare DNS for `tbeolet.com`: `CNAME` record, name = `blog`, target =
   `tristanbeolet.github.io`, proxy status **DNS only** (grey cloud).
3. GitHub repo → Settings → Pages → Custom domain: `blog.tbeolet.com`, then tick
   **Enforce HTTPS** once the certificate is issued.
4. GitHub account → Settings → Pages → Add a domain: verify it with the TXT record
   GitHub provides, added in Cloudflare.
5. If the Cloudflare proxy (orange cloud) is ever enabled, SSL/TLS mode must be **Full**,
   never Flexible (Flexible causes redirect loops).

## Writing conventions (for future sessions)

- New post: `hugo new content posts/<slug>.md`; URL pattern stays `/posts/<slug>/`
  (never change it once posts are shared).
- Each post lives as a single Markdown file or a page bundle folder with its images.
- Front matter: `title`, `date`, `tags`, `draft`, optional `description` (used for SEO).
- Multi-part posts: same `series: ["<name>"]` in each, plus a short `linkTitle`; parts are ordered by
  date. Series title and description go in `content/series/<urlized-name>/_index.md`.
- Prefer standard Markdown; use Hugo shortcodes only when they add real value.
- Link the code repo for any project-based post.
- Preview locally with `hugo server -D`.
