---
layout: page
title: submitting
permalink: /submitting/
description:
nav: false
nav_order: 3
---

## Table of contents
- [Table of contents](#table-of-contents)
- [How to Submit to the GRaM Blogpost Track](#how-to-submit-to-the-gram-blogpost-track)
- [Review model and anonymity](#review-model-and-anonymity)
- [Submission template and infrastructure](#submission-template-and-infrastructure)
- [Submission workflow (recommended)](#submission-workflow-recommended)
  - [1. Fork the repository](#1-fork-the-repository)
  - [2. Prepare your submission (strictly limited scope)](#2-prepare-your-submission-strictly-limited-scope)
  - [3. Post format](#3-post-format)
- [Local preview (optional)](#local-preview-optional)
- [Submitting your pull request](#submitting-your-pull-request)
- [Camera-ready submission](#camera-ready-submission)
- [Further questions](#further-questions)


## How to Submit to the GRaM Blogpost Track
The GRaM Blogpost–Tutorial track follows an **open, GitHub-based submission process** while preserving a double-blind review standard. Submissions are made as pull requests to a public repository and reviewed exclusively via the deployed website.

---

## Review model and anonymity

Submissions **must be anonymized for review**.  
Authors submit their blog posts via a **GitHub pull request** to the [GRaM blog 2026 Repo](https://github.com/gram-blogposts/2026).

Reviewers are instructed **not to inspect git history or repository metadata**, which may reveal author identity. This process is no less double-blind than reviewing papers already available on public preprint servers, a standard practice in machine learning.

Authors who require stricter anonymity may submit from a **new GitHub account without identifying information**, used exclusively for this track.

---

## Submission template and infrastructure

The GRaM blog uses the **al-folio** Jekyll template, with automated builds handled by **GitHub Actions**.

Submissions are validated automatically.  
⚠️ Any deviation from the required file structure will cause the submission to be rejected, so please follow the instructions carefully.

---


## Submission workflow (recommended)

### 1. Fork the repository

Fork the [GRaM blog 2026 Repo](https://github.com/gram-blogposts/2026) repository to your GitHub account.
Do **not** rename the repository and do **not** modify site configuration files.


### 2. Prepare your submission (strictly limited scope)

You must create **exactly one anonymized blog post**, introducing **only** the files listed below.

```text
_posts/
  YYYY-MM-DD-[submission-name].md

assets/
  bibliography/
    YYYY-MM-DD-[submission-name].bib
  img/
    YYYY-MM-DD-[submission-name]/
  html/
    YYYY-MM-DD-[submission-name]/   (optional)
```

The `YYYY-MM-DD-[submission-name]` identifier **must be identical across all files and folders**.
No other files may be added or modified.


### 3. Post format

Each post must:

- Use the `distill` template  
- Include a clear title  
- Include a short abstract in the `description` field  
  (2–3 sentences, **no LaTeX, no links**)  
- Include a table of contents  
- List authors as `Anonymous` during review  

Author information must be added **only after acceptance**, for the camera-ready version.

---

## Local preview (optional)

Authors may preview the site locally before submitting.

If you have Ruby and Bundler installed:

```bash
git clone https://github.com/<your-username>/2026.git
cd 2026

bundle install

bundle exec jekyll serve
```

The site will be available at:  `http://127.0.0.1:4000/`

---

## Submitting your pull request

1. Double-check that the post is fully anonymized.
2. Commit **only** the allowed files listed above.
3. Push your changes to your fork.
4. Open a pull request to the **main branch** of the GRaM Blog 2026 repository.
5. The **pull request title must exactly match the submission name**.

All updates must be made by **editing the same pull request**.  
Do not open new pull requests for revisions.

---

## Camera-ready submission

Upon acceptance, authors must:

- De-anonymize the post  
- Add author names, affiliations, and acknowledgements  
- Apply any requested editorial or formatting changes  

Detailed camera-ready instructions will be provided with acceptance notifications.

---

## Further questions

The GRaM Blogpost track is inspired by the ICLR Blogpost track.  
If you encounter technical issues with building or modifying the site, consulting their documentation may be helpful.

For questions, contact the organizers:

- organizers [at] gram-workshop [dot] org  
- manuel.lecha [at] iit.it

