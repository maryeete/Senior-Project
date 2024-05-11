import csv
import requests
import time

def fetch_products(url):
    # Fetch data from Rainforest API
    params = {
        'api_key': 'A574482E8F3B4450A249A3D1011536B1',
        'type': 'bestsellers',
        'url': url
    }
    api_result = requests.get('https://api.rainforestapi.com/request', params)
    response_json = api_result.json()
    products = response_json.get('bestsellers', [])

    updated_products = []
    for product in products:
        # Extract ASIN from the product data
        asin = product.get('asin')

        # Fetch category information using the ASIN
        category_info = fetch_category_info_with_retry(asin)

        # Update the product data with category information
        product['category_id'] = category_info.get('category_id', '')
        product['category_name'] = category_info.get('category_name', '')

        # Fetch reviews for the current product
        reviews = fetch_reviews(asin)
        
        # Check if reviews fetching was successful
        if reviews is not None:
            print("These are the reviews for:", reviews)
            
            # Prepare the reviews string
            review_text = ""
            review_ids = []

            for review in reviews:
                if 'body' in review:
                    review_text += f"Rating: {review['rating']}, Review: {review['body']}\n"
                    review_ids.append(review['id'])
                    
                else:
                    # If 'body' key does not exist, provide a placeholder
                    review_text += f"Rating: {review.get('rating', '')}, Review: [No review text available]\n"
                    review_ids.append(review.get('id', ''))      

            # Join review IDs into a comma-separated string
            review_ids_str = ', '.join(review_ids)

            # Add reviews and review IDs to the product data
            product['reviews'] = review_text
            product['review_ids'] = review_ids_str
        else:
            # If reviews fetching failed, set empty values for reviews and review IDs
            product['reviews'] = ''
            product['review_ids'] = ''

        #updated_products.append(product)
        write_product_to_csv(file, product)

    return updated_products



def fetch_category_info_with_retry(asin, max_retries=3, backoff_factor=2):
    retries = 0
    while retries < max_retries:
        try:
            print("Attempting to fetch category info for :", asin)
    
            params = {
                'api_key': 'A574482E8F3B4450A249A3D1011536B1',
                'amazon_domain': 'amazon.com',
                'asin': asin,
                'type': 'product',
            }
    
            response = requests.get('https://api.rainforestapi.com/request', params=params) 
            product_info = response.json().get("product", {})
            category_info = product_info.get("categories", [{}])[-1]  # Last category    
            category_id = category_info.get("category_id", "")
            category_name = category_info.get("name", "")

            # If fetching category info fails, use categories_flat
            if not category_name:
                category_info_flat = product_info.get("categories_flat", "")
                category_name = category_info_flat.split(" > ")[-1]
                category_id = ""

            return {'category_id': category_id, 'category_name': category_name}
        
        except (requests.exceptions.RequestException, ValueError) as e:
            print("Error fetching category info:", e)
            retries += 1
            if retries < max_retries:
                delay = backoff_factor ** retries
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries exceeded. Skipping this product.")
                return {'category_id': '', 'category_name': ''}

def fetch_reviews(asin):
    print("Attempting to fetch reviews for :", asin)
    
    params = {
        'api_key': 'A574482E8F3B4450A249A3D1011536B1',
        'type': 'reviews',
        'amazon_domain': 'amazon.com',
        'asin': asin,
        'review_stars': 'all_stars',
        'reviewer_type': 'verified_purchase',
        'sort_by': 'most_recent',
        'page': '1',
        'max_page': '3'
    }
    try:
        response = requests.get('https://api.rainforestapi.com/request', params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors
        reviews_data = response.json().get("reviews", [])
        return reviews_data
    except requests.exceptions.RequestException as e:
        print("Error fetching reviews:", e)
        return None
    except ValueError as ve:
        print("Error decoding JSON:", ve)
        return None

def write_product_to_csv(file, product):
    writer = csv.DictWriter(file, fieldnames=[
        'Product Name', 'Product ID', 'Link', 'Rating', 'Total Reviews', 
        'Price', 'Category Name', 'Category ID', 'Reviews', 'ReviewID'
    ])

    #for product in products:
        # Fetch category information using the ASIN
    category_info = fetch_category_info_with_retry(product['asin'])

    # Update the product data with category information
    product['category_id'] = category_info.get('category_id', '')
    product['category_name'] = category_info.get('category_name', '')
        
    writer.writerow({
        'Product Name': f'{product.get("title", "")}',
        'Product ID': product.get('asin', ''),
        'Link': product.get('link', ''),
        'Rating': product.get('rating', ''),
        'Total Reviews': product.get('ratings_total', ''),
        'Price': product.get('price', {}).get('raw', ''),
        'Category Name': product.get('category_name', ''),
        'Category ID': product.get('category_id', ''),
        'Reviews': product.get('reviews', ''),
        'ReviewID': product.get('review_ids', '')
    })

# Open CSV file in append mode
with open('Amazon Products Data.csv', mode='a', newline='', encoding='utf-8') as file:
    # Initialize the last row index
    last_row_index = file.tell()

    # Write header row if the file is empty
    if last_row_index == 0:
        writer = csv.DictWriter(file, fieldnames=[
            'Product Name', 'Product ID', 'Link', 'Rating', 'Total Reviews', 
            'Price', 'Category Name', 'Category ID', 'Reviews', 'ReviewID'
        ])
        writer.writeheader()

    # Iterate over each URL in the library
    url_library = [
        'https://www.amazon.com/Best-Sellers-Computers-Accessories/zgbs/pc/ref=zg_bs_unv_pc_1_516866_2',
        'https://www.amazon.com/Best-Sellers-Computers-Accessories/zgbs/pc/ref=zg_bs_pg_2_pc?_encoding=UTF8&pg=2',
        'https://www.amazon.com/Best-Sellers-Home-Kitchen/zgbs/home-garden/ref=zg_bs_pg_1_home-garden?_encoding=UTF8&pg=1',
        'https://www.amazon.com/Best-Sellers-Home-Kitchen/zgbs/home-garden/ref=zg_bs_pg_2_home-garden?_encoding=UTF8&pg=2',
        'https://www.amazon.com/Best-Sellers-Office-Products/zgbs/office-products/ref=zg_bs_nav_office-products_0',
        #'https://www.amazon.com/Best-Sellers-Office-Products/zgbs/office-products/ref=zg_bs_pg_2_office-products?_encoding=UTF8&pg=2'
        # Add more URLs here
    ]

    for url in url_library:
        fetch_products(url)
        #write_to_csv(file, products)

print("CSV file 'Amazon Products Data.csv' has been updated successfully!")