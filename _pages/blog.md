---
layout: page
title: accepted posts
permalink: /blog/
nav: true
nav_order: 2
pagination:
  enabled: true
  collection: posts
  permalink: /page/:num/
  per_page: 15
  sort_field: date
  sort_reverse: true
---

<div class="post">

{% if page.display_categories %}
  <ul class="post-categories">
    {% for category in page.display_categories %}
      <li><a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a></li>
    {% endfor %}
  </ul>
{% endif %}

<ul class="post-list">
  {% assign posts = paginator.posts %}
  {% for post in posts %}
    <li>
      <h3>
        <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h3>
      <p class="post-meta">{% if post.authors %}{% for author in post.authors %}{{ author.name }}{% unless forloop.last %}, {% endunless %}{% endfor %}{% endif %}</p>
      <p class="post-description">{{ post.description | truncate: 180 }}</p>
    </li>
  {% endfor %}
</ul>

{% include pagination.liquid %}

</div>
