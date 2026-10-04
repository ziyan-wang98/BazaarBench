"""Marketplace listing ingestion for scale-up worlds.

Supported public-data shapes are intentionally permissive:

* Mercari/Kaggle-like CSV: ``name``, ``item_description``,
  ``category_name``, ``brand_name``, ``item_condition_id``, ``price``.
* PromptCloud eBay-like CSV: ``Title``, ``Price``, ``Manufacturer``,
  ``Model Name``, ``Model Num``, ``Average Rating``, ``Stock``.
"""
from __future__ import annotations

import csv
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bazaar.agents.catalog import ITEM_CATALOG

_DEFAULT_SOURCE = "catalog_fallback"


@dataclass(frozen=True)
class MarketplaceItem:
    unique_id: str | None
    category: str
    title: str
    description: str
    price_cents: int
    condition: str
    brand: str | None = None
    model_name: str | None = None
    rating: float | None = None
    num_reviews: int | None = None
    seller_rating: float | None = None
    seller_num_reviews: int | None = None
    star_counts: dict[str, int] | None = None
    crawl_timestamp: str | None = None
    stock: str | None = None
    page_url: str | None = None
    source: str = _DEFAULT_SOURCE

    def to_inventory_item(self) -> dict[str, Any]:
        data = {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "asking_price_cents": self.price_cents,
            "condition": self.condition,
            "source": "marketplace_dataset",
        }
        if self.brand:
            data["brand"] = self.brand
        if self.model_name:
            data["model_name"] = self.model_name
        attrs: dict[str, Any] = {}
        if self.rating is not None:
            attrs["rating"] = self.rating
        if self.num_reviews is not None:
            attrs["num_reviews"] = self.num_reviews
        if self.seller_rating is not None:
            attrs["seller_rating"] = self.seller_rating
        if self.seller_num_reviews is not None:
            attrs["seller_num_reviews"] = self.seller_num_reviews
        if self.star_counts:
            attrs["star_counts"] = self.star_counts
        if self.unique_id:
            attrs["unique_id"] = self.unique_id
        if self.crawl_timestamp:
            attrs["crawl_timestamp"] = self.crawl_timestamp
        if self.stock:
            attrs["stock"] = self.stock
        if self.page_url:
            attrs["page_url"] = self.page_url
        if attrs:
            data["dataset_attrs"] = attrs
        return data

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def catalog_items() -> list[MarketplaceItem]:
    """Return the built-in catalog as normalized marketplace items."""
    out: list[MarketplaceItem] = []
    for category, rows in ITEM_CATALOG.items():
        for title, description, low, high in rows:
            out.append(
                MarketplaceItem(
                    unique_id=None,
                    category=category,
                    title=title,
                    description=description,
                    price_cents=(int(low) + int(high)) // 2,
                    condition="good",
                    source=_DEFAULT_SOURCE,
                )
            )
    return out


def load_marketplace_items(
    csv_path: str | Path | None = None,
    *,
    limit: int = 10_000,
    sample_fraction: float | None = None,
    seed: int = 7,
    min_price_cents: int = 100,
    max_price_cents: int | None = 500_000,
) -> list[MarketplaceItem]:
    """Load normalized marketplace rows, falling back to the repo catalog.

    ``limit`` uses reservoir sampling so large public CSVs do not have to
    be held fully in memory. If ``csv_path`` is absent or yields no valid
    rows, the hand-authored BazaarBench catalog is returned.

    ``min_price_cents`` / ``max_price_cents`` filter raw-data outliers — the
    public Kaggle eBay snapshot has ~7% of rows with prices > $100K (e.g.
    iPhone 7 listed at $15,751.37) that would corrupt downstream cold-start
    inventory and rollout listing prices. Default cap is $5,000 which keeps
    legitimate big-ticket items but drops the obvious garbage. Pass
    ``max_price_cents=None`` to disable filtering.
    """
    if csv_path is None:
        return catalog_items()
    path = Path(csv_path)
    if not path.exists():
        return catalog_items()
    if sample_fraction is not None and not 0 < sample_fraction <= 1:
        raise ValueError("sample_fraction must be in (0, 1]")
    rng = random.Random(seed)
    reservoir: list[MarketplaceItem] = []
    seen = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if sample_fraction is not None and rng.random() > sample_fraction:
                continue
            item = normalize_marketplace_row(raw, source=path.name)
            if item is None:
                continue
            if item.price_cents < min_price_cents:
                continue
            if max_price_cents is not None and item.price_cents > max_price_cents:
                continue
            seen += 1
            if limit <= 0 or len(reservoir) < limit:
                reservoir.append(item)
                continue
            idx = rng.randint(0, seen - 1)
            if idx < limit:
                reservoir[idx] = item
    return reservoir or catalog_items()


def normalize_marketplace_row(
    row: dict[str, Any],
    *,
    source: str = "marketplace_csv",
) -> MarketplaceItem | None:
    """Normalize one public marketplace row into BazaarBench shape."""
    title = _first_text(
        row,
        "name",
        "title",
        "listing_title",
        "Title",
        "Product Name",
        "SKU Name",
        "SKU_Name",
    )
    if not title:
        return None
    brand = _first_text(
        row,
        "brand_name",
        "brand",
        "Brand Name",
        "Brand",
        "Manufacturer",
        "manufacturer",
    ) or None
    model_name = _first_text(
        row,
        "model_name",
        "Model Name",
        "Model Num",
        "Model Number",
        "model_num",
        "Sku",
        "SKU",
        "sku",
    ) or None
    description = _first_text(
        row,
        "item_description",
        "description",
        "desc",
        "Product Description",
        "About Product",
        "Product Specification",
        "Technical Details",
        default=_synthetic_description(
            title=title,
            brand=brand,
            model_name=model_name,
            stock=_first_text(row, "Stock", "stock"),
            rating=_first_text(row, "Average Rating", "average_rating"),
        ),
    )
    if description.lower() in {"no description yet", "nan", "none"}:
        description = _synthetic_description(
            title=title,
            brand=brand,
            model_name=model_name,
            stock=_first_text(row, "Stock", "stock"),
            rating=_first_text(row, "Average Rating", "average_rating"),
        )
    raw_category = _first_text(
        row,
        "category_name",
        "category",
        "category_path",
        "Category",
        "Category_1",
        "Sub_Category",
        "Sub Category",
        "Sector",
        "Sub_Sector",
        "Color Category",
    )
    category = _map_category(" ".join([title, description, brand or "", raw_category]))
    price_cents = _parse_price_cents(
        _first_text(
            row,
            "price_cents",
            "price",
            "listing_price",
            "Price",
            "Selling Price",
            "List Price",
            "Price_In_Local_USD",
            "Price_In_Local",
        )
    )
    if price_cents is None or price_cents <= 0:
        return None
    star_counts = _parse_star_counts(row)
    return MarketplaceItem(
        unique_id=_clean_text(
            _first_text(row, "Uniq Id", "uniq_id", "id", "product_id"),
            max_len=120,
        ) or None,
        category=category,
        title=_clean_text(title, max_len=120),
        description=_clean_text(description, max_len=400),
        price_cents=price_cents,
        condition=_normalize_condition(
            _first_text(
                row,
                "condition",
                "item_condition_id",
                "condition_id",
                "Condition",
                "Stock",
                "stock",
            )
        ),
        brand=_clean_text(brand, max_len=80) if brand else None,
        model_name=_clean_text(model_name, max_len=100) if model_name else None,
        rating=_parse_float(_first_text(row, "Average Rating", "average_rating")),
        num_reviews=_parse_int(_first_text(
            row,
            "Num Of Reviews",
            "Number Of Ratings",
            "num_reviews",
            "reviews",
        )),
        seller_rating=_parse_float(_first_text(
            row,
            "Seller Rating",
            "seller_rating",
            "SellerRating",
            "Seller Score",
        )),
        seller_num_reviews=_parse_int(_first_text(
            row,
            "Seller Num Of Reviews",
            "Seller Number Of Reviews",
            "seller_num_reviews",
            "Seller Reviews",
        )),
        star_counts=star_counts or None,
        crawl_timestamp=_clean_text(
            _first_text(row, "Crawl Timestamp", "crawl_timestamp", "timestamp"),
            max_len=80,
        ) or None,
        stock=_clean_text(_first_text(row, "Stock", "stock"), max_len=80) or None,
        page_url=_clean_text(_first_text(row, "Pageurl", "Product Url", "url"), max_len=240) or None,
        source=source,
    )


def _first_text(
    row: dict[str, Any],
    *keys: str,
    default: str = "",
) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _clean_text(text: str | None, *, max_len: int) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return cleaned[:max_len].strip()


def _parse_price_cents(raw: str) -> int | None:
    text = str(raw or "").strip().replace("$", "").replace(",", "")
    if not text:
        return None
    range_match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$", text)
    if range_match:
        lo = float(range_match.group(1))
        hi = float(range_match.group(2))
        return int(round(((lo + hi) / 2) * 100))
    try:
        value = float(text)
    except ValueError:
        return None
    if value <= 0:
        return None
    if value > 10_000 and "." not in text:
        return int(value)
    return int(round(value * 100))


def _normalize_condition(raw: str) -> str:
    text = str(raw or "").strip().lower()
    if text in {"1", "new", "new with tags", "brand new", "in stock"}:
        return "new"
    if text in {"2", "like new", "excellent", "open box"}:
        return "like_new"
    if text in {"3", "good", "used"}:
        return "good"
    if text in {"4", "fair", "acceptable"}:
        return "fair"
    if text in {"5", "poor", "salvage", "for parts"}:
        return "poor"
    return "good"


def _parse_float(raw: str) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(raw: str) -> int | None:
    text = str(raw or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _parse_star_counts(row: dict[str, Any]) -> dict[str, int]:
    """Parse PromptCloud star-histogram columns when present."""
    mapping = {
        "five": ("Five Star", "five_star", "5 Star", "5_star"),
        "four": ("Four Star", "four_star", "4 Star", "4_star"),
        "three": ("Three Star", "three_star", "3 Star", "3_star"),
        "two": ("Two Star", "two_star", "2 Star", "2_star"),
        "one": ("One Star", "one_star", "1 Star", "1_star"),
    }
    out: dict[str, int] = {}
    for label, keys in mapping.items():
        value = _parse_int(_first_text(row, *keys))
        if value is not None:
            out[label] = value
    return out


def _synthetic_description(
    *,
    title: str,
    brand: str | None,
    model_name: str | None,
    stock: str,
    rating: str,
) -> str:
    parts = [f"Observed marketplace listing for {title}."]
    if brand:
        parts.append(f"Brand/manufacturer: {brand}.")
    if model_name:
        parts.append(f"Model: {model_name}.")
    if stock:
        parts.append(f"Stock field: {stock}.")
    if rating:
        parts.append(f"Average rating: {rating}.")
    return " ".join(parts)


_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("electronics-phones", ("iphone", "smartphone", "cell phone", "galaxy")),
    ("electronics-laptops", ("laptop", "macbook", "thinkpad", "chromebook")),
    ("electronics-tablets", ("tablet", "ipad", "kindle fire")),
    ("electronics-cameras", ("camera", "lens", "drone", "gopro", "canon", "sony a7")),
    ("electronics-cameras", ("dslr", "mirrorless", "camcorder")),
    ("electronics-audio", ("headphone", "airpods", "speaker", "stereo", "vinyl")),
    ("electronics-gaming", ("playstation", "xbox", "nintendo", "switch", "gaming")),
    ("collectibles-tcg", ("pokemon", "magic the gathering", "mtg", "yugioh", "tcg")),
    ("collectibles-figures", ("lego", "funko", "figure", "bearbrick", "hot toys")),
    ("collectibles-sports", ("signed jersey", "rookie", "sports card", "coa")),
    ("jewelry", ("watch", "bracelet", "necklace", "ring", "gold", "diamond")),
    ("vehicles-cars", ("car ", "sedan", "truck", "vespa", "miles")),
    ("vehicles-bikes", ("bike", "bicycle", "scooter", "trek", "specialized")),
    ("furniture", ("sofa", "desk", "chair", "bed frame", "dresser", "table")),
    ("home-goods", ("kitchen", "mixer", "espresso", "blender", "cookware", "home")),
    ("sporting-goods", ("fitness", "camping", "treadmill", "dumbbell", "kayak")),
    ("tools", ("tool", "drill", "saw", "woodworking")),
    ("books", ("book", "novel", "textbook")),
    ("clothing", ("clothing", "shirt", "dress", "jacket", "sneaker", "shoe")),
    ("kids", ("baby", "kids", "stroller", "toy")),
    ("garden", ("garden", "plant", "lawn")),
    ("musical-instruments", ("guitar", "piano", "keyboard", "instrument")),
    ("pets", ("pet", "dog", "cat", "aquarium")),
)


def _map_category(text: str) -> str:
    haystack = f" {text.lower().replace('/', ' ')} "
    for category, needles in _CATEGORY_RULES:
        if any(needle in haystack for needle in needles):
            return category
    return "home-goods"
