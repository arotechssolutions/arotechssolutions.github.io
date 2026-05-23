""" Product Image Clustering Script """

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np

# ── Simulated scraped product data ───────────────────────────────────────────
# Images from picsum.photos (deterministic, free, no auth needed)
PRODUCTS = [
    # Electronics
    {"title":"Wireless Noise-Cancelling Headphones","price":"$249.99","image":"https://picsum.photos/seed/headphones1/300/300","description":"Over-ear Bluetooth headphones with 30-hour battery, active noise cancellation and premium sound quality for music and calls."},
    {"title":"True Wireless Earbuds","price":"$89.99","image":"https://picsum.photos/seed/earbuds2/300/300","description":"Compact wireless earbuds with deep bass, touch controls and sweat-resistant design perfect for workouts and commuting."},
    {"title":"Bluetooth Speaker","price":"$59.99","image":"https://picsum.photos/seed/speaker3/300/300","description":"Portable waterproof Bluetooth speaker delivering 360 degree stereo sound with 12-hour playtime and rugged outdoor build."},
    {"title":"Smartwatch Fitness Tracker","price":"$199.99","image":"https://picsum.photos/seed/watch4/300/300","description":"Advanced smartwatch with heart rate monitor, GPS, sleep tracking, and smartphone notifications on a vibrant AMOLED display."},
    {"title":"4K Ultra HD Monitor","price":"$449.99","image":"https://picsum.photos/seed/monitor5/300/300","description":"27-inch 4K IPS display with 144Hz refresh rate, USB-C connectivity and factory-calibrated wide color gamut for professionals."},
    {"title":"Mechanical Gaming Keyboard","price":"$129.99","image":"https://picsum.photos/seed/keyboard6/300/300","description":"Full-size mechanical keyboard with Cherry MX switches, per-key RGB backlighting, and aluminum frame for competitive gaming."},
    {"title":"Wireless Gaming Mouse","price":"$79.99","image":"https://picsum.photos/seed/mouse7/300/300","description":"High-precision 25K DPI wireless gaming mouse with programmable buttons, ergonomic grip and ultra-low latency receiver."},
    {"title":"USB-C Laptop Hub","price":"$49.99","image":"https://picsum.photos/seed/hub8/300/300","description":"7-in-1 USB-C hub with HDMI 4K output, 3 USB ports, SD card reader, and 100W power delivery for modern laptops."},
    {"title":"Portable SSD 1TB","price":"$109.99","image":"https://picsum.photos/seed/ssd9/300/300","description":"Ultra-fast portable solid state drive with 1050MB/s read speed, shock-resistant aluminum enclosure and USB-C interface."},
    {"title":"Smart Home Security Camera","price":"$69.99","image":"https://picsum.photos/seed/cam10/300/300","description":"1080p indoor smart camera with two-way audio, motion detection alerts, night vision and cloud storage subscription."},
    # Clothing
    {"title":"Men's Slim-Fit Chinos","price":"$59.99","image":"https://picsum.photos/seed/chinos11/300/300","description":"Stretch cotton slim-fit chino trousers in classic navy, ideal for smart-casual office wear or weekend outings."},
    {"title":"Women's Floral Summer Dress","price":"$44.99","image":"https://picsum.photos/seed/dress12/300/300","description":"Lightweight floral print midi dress with adjustable straps and a flowy silhouette, perfect for warm-weather occasions."},
    {"title":"Men's Wool Blend Overcoat","price":"$189.99","image":"https://picsum.photos/seed/coat13/300/300","description":"Tailored wool-blend overcoat with notched lapels, double-button closure and a warm inner lining for cold seasons."},
    {"title":"Women's Yoga Leggings","price":"$39.99","image":"https://picsum.photos/seed/leggings14/300/300","description":"High-waist compression yoga leggings with moisture-wicking fabric, hidden pocket and four-way stretch for all activities."},
    {"title":"Men's Graphic T-Shirt","price":"$24.99","image":"https://picsum.photos/seed/tshirt15/300/300","description":"Soft 100% cotton crewneck T-shirt featuring bold original graphic print, available in multiple colors and relaxed fit."},
    {"title":"Women's Denim Jacket","price":"$74.99","image":"https://picsum.photos/seed/denim16/300/300","description":"Classic washed denim jacket with button-front closure, chest pockets and rolled cuffs for a vintage-inspired street look."},
    {"title":"Men's Running Sneakers","price":"$119.99","image":"https://picsum.photos/seed/sneakers17/300/300","description":"Lightweight mesh running sneakers with responsive foam midsole, breathable upper and anti-slip rubber outsole."},
    {"title":"Women's Block Heel Boots","price":"$99.99","image":"https://picsum.photos/seed/boots18/300/300","description":"Ankle-height block heel boots in genuine leather with side zip, cushioned insole and stacked heel for everyday elegance."},
    # Kitchen & Home
    {"title":"Stainless Steel Air Fryer","price":"$89.99","image":"https://picsum.photos/seed/airfryer19/300/300","description":"5.8-quart air fryer with digital touchscreen, 8 cooking presets and rapid hot-air circulation for crispy oil-free cooking."},
    {"title":"Non-Stick Cookware Set","price":"$119.99","image":"https://picsum.photos/seed/cookware20/300/300","description":"10-piece ceramic non-stick cookware set including saucepans, frying pans and a stockpot with tempered glass lids."},
    {"title":"Espresso Coffee Machine","price":"$299.99","image":"https://picsum.photos/seed/espresso21/300/300","description":"15-bar pump espresso machine with built-in milk frother, programmable settings and stainless steel thermoblock heating."},
    {"title":"Electric Stand Mixer","price":"$249.99","image":"https://picsum.photos/seed/mixer22/300/300","description":"5-quart tilt-head stand mixer with 10 speeds, planetary mixing action and stainless steel bowl for baking enthusiasts."},
    {"title":"Bamboo Cutting Board Set","price":"$34.99","image":"https://picsum.photos/seed/cutting23/300/300","description":"Set of 3 eco-friendly bamboo cutting boards in different sizes with juice grooves and non-slip feet for safe chopping."},
    {"title":"Cast Iron Skillet","price":"$44.99","image":"https://picsum.photos/seed/skillet24/300/300","description":"Pre-seasoned 12-inch cast iron skillet with helper handle, compatible with all stovetops and oven-safe up to 500°F."},
    {"title":"Vacuum Insulated Water Bottle","price":"$29.99","image":"https://picsum.photos/seed/bottle25/300/300","description":"32oz double-wall vacuum insulated stainless steel bottle keeping drinks cold 24 hours and hot 12 hours with leak-proof lid."},
    {"title":"Smart Thermostat","price":"$149.99","image":"https://picsum.photos/seed/thermostat26/300/300","description":"Wi-Fi programmable smart thermostat with learning algorithm, energy reports and remote control via smartphone app."},
    # Sports & Outdoors
    {"title":"Adjustable Dumbbells Set","price":"$229.99","image":"https://picsum.photos/seed/dumbbells27/300/300","description":"Space-saving adjustable dumbbell set ranging from 5 to 52.5 lbs per dumbbell with quick-change weight selector dial."},
    {"title":"Yoga Mat with Strap","price":"$29.99","image":"https://picsum.photos/seed/yogamat28/300/300","description":"Thick 6mm non-slip TPE yoga mat with alignment lines, carrying strap and moisture-resistant surface for home workouts."},
    {"title":"Mountain Bike Helmet","price":"$79.99","image":"https://picsum.photos/seed/helmet29/300/300","description":"Lightweight MIPS mountain bike helmet with 20 vents, adjustable fit dial and removable visor for trail riding safety."},
    {"title":"Camping Tent 4-Person","price":"$159.99","image":"https://picsum.photos/seed/tent30/300/300","description":"Four-person instant setup camping tent with waterproof rainfly, mesh ventilation panels and gear loft for family adventures."},
    {"title":"Trekking Poles Set","price":"$54.99","image":"https://picsum.photos/seed/poles31/300/300","description":"Collapsible carbon fiber trekking poles with ergonomic cork grips, quick-lock system and interchangeable rubber tips."},
    {"title":"Resistance Bands Kit","price":"$24.99","image":"https://picsum.photos/seed/bands32/300/300","description":"Set of 5 latex resistance bands in varying strengths for strength training, physical therapy and full-body workouts at home."},
    # Books
    {"title":"The Art of Computer Programming","price":"$79.99","image":"https://picsum.photos/seed/book33/300/300","description":"Comprehensive multi-volume reference covering algorithms, data structures and programming techniques by Donald Knuth."},
    {"title":"Atomic Habits","price":"$17.99","image":"https://picsum.photos/seed/book34/300/300","description":"Practical guide to building good habits and breaking bad ones through tiny behavioral changes with proven scientific methods."},
    {"title":"Deep Learning Textbook","price":"$64.99","image":"https://picsum.photos/seed/book35/300/300","description":"Comprehensive textbook on deep learning covering neural networks, backpropagation, convolutional and recurrent architectures."},
    {"title":"The Pragmatic Programmer","price":"$49.99","image":"https://picsum.photos/seed/book36/300/300","description":"Classic software engineering book with timeless advice on coding practices, debugging and building maintainable software."},
    {"title":"Sapiens: A Brief History","price":"$15.99","image":"https://picsum.photos/seed/book37/300/300","description":"Fascinating exploration of human history from ancient foragers to modern civilization covering culture, science and society."},
    {"title":"Python Crash Course","price":"$35.99","image":"https://picsum.photos/seed/book38/300/300","description":"Hands-on beginner guide to Python programming covering variables, loops, functions and building real-world projects."},
]

def cluster_products(products, n_clusters=6):
    print(f"Clustering {len(products)} products into {n_clusters} clusters via TF-IDF + K-Means...")
    corpus = [p["description"] + " " + p["title"] for p in products]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=500, ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus)
    X_norm = normalize(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    labels = km.fit_predict(X_norm)

    # Generate cluster label from top TF-IDF centroid terms
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_names = {}
    for i in range(n_clusters):
        top = [terms[idx] for idx in order_centroids[i, :3]]
        cluster_names[i] = " · ".join(t.replace("_", " ").title() for t in top)

    for i, p in enumerate(products):
        p["cluster"] = int(labels[i])
        p["cluster_name"] = cluster_names[int(labels[i])]

    return products, cluster_names


def main():
    products, cluster_names = cluster_products(PRODUCTS, n_clusters=6)
    output = {
        "cluster_names": {str(k): v for k, v in cluster_names.items()},
        "products": products
    }
    with open("/home/claude/clusters.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nDone! Saved clusters.json")
    from collections import Counter
    counts = Counter(p["cluster"] for p in products)
    for cid, name in cluster_names.items():
        print(f"  Cluster {cid} [{name}]: {counts[cid]} products")

if __name__ == "__main__":
    main()
