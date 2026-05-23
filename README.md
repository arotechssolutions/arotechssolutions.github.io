# Product Image Clustering by STANY GANYANI R204442S HAI
**Web Mining & Recommender Systems — Assignment**
> Algorithm: TF-IDF Vectorization + K-Means Clustering

---

## Overview

For this assignment I chose Option 3 — clustering product images according to the similarity of the text descriptions that describe them. The idea was: Similar products are described with similar vocabulary, so by analysing those descriptions mathematically I could group related items together automatically, without manually labelling a single one.

---

## What the Project Does

The project has two main parts that work together: a Python script that handles all the data collection and clustering work, and an HTML web page that presents the results visually.

### Step 1 — Collecting the Product Data

I started by putting together a dataset of 38 products spread across five real-world categories: Electronics, Clothing, Kitchen & Home, Sports & Outdoors, and Books. Each product has a title, price, image URL, and a text description — exactly the kind of information you would find if you scraped a site like Amazon or any e-commerce store.

### Step 2 — Turning Text into Numbers (TF-IDF)

Computers cannot directly compare words, so the first real processing step is converting each product's description into a numerical vector using a technique called **TF-IDF (Term Frequency–Inverse Document Frequency)**.

Here is what that means in plain terms:

- **Term Frequency** measures how often a word appears in a single product description. A word that appears many times is probably important to that product.
- **Inverse Document Frequency** penalises words that appear across almost every product — words like *with*, *and*, or *for* tell us nothing useful because they appear everywhere. TF-IDF gives higher weight to words that are distinctive to a small number of products.

The result is a 500-dimensional vector for each product. I also included bigrams (two-word combinations like *stainless steel* or *noise cancellation*) so the model can pick up on meaningful phrases rather than just single words.

### Step 3 — Clustering with K-Means

Once every product is represented as a vector, I used the **K-Means algorithm** to group them into 6 clusters. K-Means works by:

1. Picking 6 random starting points (centroids) in the vector space
2. Assigning every product to its nearest centroid based on cosine similarity
3. Recalculating each centroid as the average of all products assigned to it
4. Repeating this process until the assignments stop changing

I normalised the vectors first so that cosine similarity which measures the angle between vectors rather than their magnitude is used as the distance metric. This is standard practice for text data because it avoids giving unfair weight to longer descriptions.

The algorithm ran with 15 different random initialisations (`n_init=15`) to avoid settling into a poor local minimum, and I set a fixed random seed (`random_state=42`) so the results are reproducible every time the script is run.

### Step 4 — Naming the Clusters Automatically

Once the clusters were formed, I gave each one a label by looking at the top three TF-IDF terms from the centroid of each cluster. These are the words that sit most centrally in that group, meaning they best represent what the cluster is about. The result was six meaningful clusters:

| Cluster | Label | Description |
|---------|-------|-------------|
| 0 | Kitchen & Steel | Kitchen appliances with stainless steel components |
| 1 | Tech & Gaming | Wireless devices, gaming peripherals, and electronics sets |
| 2 | Books & Learning | Educational books and learning resources |
| 3 | Portable Electronics | USB, portable, and high-resolution display devices |
| 4 | Fitness & Smart Home | Yoga equipment, smart devices, and cameras |
| 5 | Apparel & Outdoors | Clothing, footwear, and outdoor gear |

### Step 5 — Displaying the Results

The final output is a self-contained HTML web page (`product_clusters.html`) that loads all 38 products grouped visually by their cluster. Each cluster gets its own colour-coded section with a header showing the cluster name and item count. Every product card shows the image, title, description, and price.

The page also includes a live search bar so you can filter products by keyword, and filter buttons so you can isolate any single cluster at a time. Everything runs in the browser with no server needed — just open the file and it works.

---

## Project Files

```
.
├── scrape_and_cluster.py   # Python backend — data, TF-IDF, K-Means, outputs clusters.json
├── clusters.json           # Structured clustering output (38 products + cluster assignments)
└── product_clusters.html   # Self-contained web page displaying clustered products
```

| File | Purpose |
|------|---------|
| `scrape_and_cluster.py` | Builds the product dataset, runs TF-IDF vectorisation, fits K-Means, labels clusters, and writes `clusters.json` |
| `clusters.json` | The data bridge between Python and the web page — contains all products with cluster IDs and names |
| `product_clusters.html` | Front-end visualisation with colour-coded clusters, search, and filter controls |

---

## How to Run

### Option A — Open the Web Page Directly (Simplest)

The HTML file is fully self-contained. Just download `product_clusters.html` and open it in any browser. No installation, no server, no Python needed.

## Technologies Used

| Layer | Technology |
|-------|-----------|
| Web scraping | `requests` + `BeautifulSoup4` |
| Text vectorisation | `scikit-learn` TF-IDF (500 features, unigrams + bigrams) |
| Clustering algorithm | `scikit-learn` K-Means (k=6, cosine similarity) |
| Front-end | HTML, CSS, vanilla JavaScript |
| Typography | Google Fonts — Syne + DM Sans |
| Product images | [picsum.photos](https://picsum.photos) (deterministic placeholders) |
---

*Web Mining & Recommender Systems Assignment · Option 3: Product Image Clustering · K-Means + TF-IDF*
