from __future__ import annotations

from bazaar.data import load_marketplace_items, normalize_marketplace_row


def test_normalize_mercari_like_camera_row() -> None:
    item = normalize_marketplace_row({
        "name": "Canon EOS R body with battery",
        "item_description": "Used for one year, working shutter.",
        "category_name": "Electronics/Cameras",
        "brand_name": "Canon",
        "item_condition_id": "2",
        "price": "599.00",
    })

    assert item is not None
    assert item.category == "electronics-cameras"
    assert item.condition == "like_new"
    assert item.price_cents == 59_900
    assert item.brand == "Canon"


def test_load_marketplace_items_falls_back_to_catalog(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    items = load_marketplace_items(missing, limit=10, seed=1)

    assert items
    assert {item.source for item in items} == {"catalog_fallback"}


def test_load_marketplace_items_reads_csv(tmp_path) -> None:
    csv_path = tmp_path / "items.csv"
    csv_path.write_text(
        "name,item_description,category_name,brand_name,item_condition_id,price\n"
        "Sony WH-1000XM5,Noise cancelling headphones,Electronics/Audio,Sony,3,220\n"
        "Bad row,no price,Other,,3,\n",
        encoding="utf-8",
    )

    items = load_marketplace_items(csv_path, limit=10, seed=1)

    assert len(items) == 1
    assert items[0].category == "electronics-audio"
    assert items[0].price_cents == 22_000
    assert items[0].source == "items.csv"


def test_normalize_promptcloud_ebay_row_enriches_attributes() -> None:
    item = normalize_marketplace_row({
        "Uniq Id": "abc",
        "Pageurl": "https://www.ebay.com/itm/example",
        "Title": "Apple iPhone 14 Pro 128GB Space Black",
        "Manufacturer": "Apple",
        "Model Name": "iPhone 14 Pro",
        "Price": "$799.99",
        "Average Rating": "4.8",
        "Number Of Ratings": "1234",
        "Stock": "In Stock",
        "Color Category": "Cell Phones & Smartphones",
    })

    assert item is not None
    assert item.category == "electronics-phones"
    assert item.brand == "Apple"
    assert item.model_name == "iPhone 14 Pro"
    assert item.price_cents == 79_999
    assert item.condition == "new"
    assert item.rating == 4.8
    assert item.num_reviews == 1234
    inventory = item.to_inventory_item()
    assert inventory["model_name"] == "iPhone 14 Pro"
    assert inventory["dataset_attrs"]["stock"] == "In Stock"
