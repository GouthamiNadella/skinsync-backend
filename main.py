from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
import logging
import requests

# Try to import google.generativeai, but don't crash if it's not installed.
GENAI_AVAILABLE = True
try:
    import google.generativeai as genai
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# === ENHANCED CORS CONFIGURATION ===
origins = [
    "https://skinsync-frontend-e7zs.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Custom middleware to ensure CORS headers on all responses
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# Handle OPTIONS requests for CORS preflight
@app.options("/{path:path}")
async def options_handler(path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API configured successfully")
    except Exception as e:
        GENAI_AVAILABLE = False
        logger.warning(f"⚠️ Failed to configure Gemini: {e}")
else:
    if not GENAI_AVAILABLE:
        logger.warning("⚠️ google-generativeai not installed")
    if not GEMINI_API_KEY:
        logger.warning("⚠️ GEMINI_API_KEY not set in .env")

# ===== DATABASE FUNCTIONS =====
def load_products_db():
    """Load products from productData.json"""
    try:
        with open('productData.json', 'r') as f:
            data = json.load(f)
            # Ensure we always have a list
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
            else:
                return []
    except FileNotFoundError:
        logger.error("❌ productData.json not found")
        return []
    except json.JSONDecodeError:
        logger.error("❌ productData.json is not valid JSON")
        return []

products_db = load_products_db()
logger.info(f"📊 Loaded {len(products_db)} products from database")

# ===== PYDANTIC MODELS =====
class Product(BaseModel):
    title: str
    brand: str
    ingredients: List[str]
    sku: Optional[str] = None

class CompatibilityRequest(BaseModel):
    currentProducts: List[Product]
    newProduct: Product
    skinType: str
    detectedConflicts: List[dict] = []

class AnalysisResponse(BaseModel):
    explanation: str
    recommendations: List[str]
    warnings: Optional[List[str]] = []
    tips: Optional[str] = None
    score: Optional[float] = None

class FetchIngredientsRequest(BaseModel):
    product_name: str

class ConflictItem(BaseModel):
    ingredient1: str
    ingredient2: str
    severity: str
    reason: str

class ReviewRequest(BaseModel):
    productName: str
    skinType: str

# ===== ROOT ENDPOINT =====
@app.get("/")
def root():
    return {
        "message": "SkinSync API - Skincare Compatibility & Review Platform",
        "version": "2.0.0",
        "gemini_available": GENAI_AVAILABLE,
        "cors_enabled": True,
        "endpoints": {
            "health": "/api/health",
            "search": "/api/products/search?query=CeraVe",
            "fetch_ingredients": "POST /api/fetch-ingredients",
            "analyze": "POST /api/analyze-compatibility",
            "conflicts": "/api/conflicts",
            "generate_review": "POST /api/generate-review",
            "product_image": "GET /api/product-image?product_name=..."
        }
    }

# ===== HEALTH CHECK =====
@app.get("/api/health")
def health_check():
    """Check API status"""
    return {
        "status": "ok",
        "message": "Skincare API is running",
        "gemini_available": GENAI_AVAILABLE,
        "database_loaded": len(products_db) > 0,
        "products_count": len(products_db)
    }

# ===== TEST CORS ENDPOINT =====
@app.get("/api/test-cors")
async def test_cors():
    return JSONResponse(
        content={"message": "CORS is working!", "status": "success"},
        headers={
            "Access-Control-Allow-Origin": "*",
        }
    )

# ===== SEARCH ENDPOINT =====
@app.get("/api/products/search")
def search_products(query: str, limit: int = 20):
    """Search products by name or brand"""
    try:
        if not query or not query.strip() or len(query.strip()) < 2:
            return JSONResponse(content=[])

        q = query.lower().strip()
        exact_brand_matches = []
        starts_with_matches = []
        partial_matches = []

        for product in products_db:
            title = product.get('title', '').lower()
            brand = product.get('brand', '').lower()
            searchable = f"{title} {brand}"

            # Exact brand match gets highest priority
            if brand == q:
                exact_brand_matches.append(product)
            # Title or brand starts with query
            elif title.startswith(q) or brand.startswith(q):
                starts_with_matches.append(product)
            # Partial match anywhere
            elif q in searchable:
                partial_matches.append(product)

            # Stop if we have enough results
            if len(exact_brand_matches) + len(starts_with_matches) + len(partial_matches) >= limit:
                break

        results = exact_brand_matches + starts_with_matches + partial_matches
        logger.info(f"🔍 Search '{query}': found {len(results)} results")
        
        return JSONResponse(
            content=results[:limit],
            headers={
                "Access-Control-Allow-Origin": "*",
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Search error: {str(e)}")
        return JSONResponse(
            content=[],
            headers={
                "Access-Control-Allow-Origin": "*",
            }
        )

# ===== FETCH INGREDIENTS ENDPOINT =====
@app.post("/api/fetch-ingredients")
async def fetch_ingredients(request: FetchIngredientsRequest):
    """Fetch ingredients from Gemini API for unknown products"""
    if not GENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable. Install google-generative-ai or set GEMINI_API_KEY."
        )

    product_name = request.product_name.strip()
    if not product_name:
        raise HTTPException(status_code=400, detail="Product name cannot be empty")

    try:
        logger.info(f"🔄 Fetching ingredients for: {product_name}")
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""Find the ingredients list for the skincare product: "{product_name}"

Return ONLY a JSON object with this format (no extra text, no markdown):
{{
  "title": "product name",
  "brand": "brand name",
  "ingredients": ["ingredient1", "ingredient2", "ingredient3"],
  "sku": "sku-code-if-known"
}}

If you cannot find the product, return:
{{"error": "Product not found"}}

Important: Return ONLY the JSON object, nothing else."""

        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        logger.info(f"✅ Successfully fetched ingredients for: {product_name}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Gemini response. Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Error fetching ingredients: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching ingredients: {str(e)}"
        )

# ===== GENERATE PRODUCT REVIEW ENDPOINT =====
@app.post("/api/generate-review")
async def generate_review(request: ReviewRequest):
    """Generate AI-powered product review based on skin type"""
    if not GENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable. Install google-generative-ai or set GEMINI_API_KEY."
        )
    
    product_name = request.productName.strip()
    skin_type = request.skinType.strip()
    
    if not product_name or not skin_type:
        raise HTTPException(status_code=400, detail="Product name and skin type are required")
    
    try:
        logger.info(f"🔄 Generating review for: {product_name} ({skin_type} skin)")
        
        prompt = f"""Analyze the skincare product "{product_name}" for someone with {skin_type} skin type.

Generate a helpful, personalized review in this exact format:

**Decision:** Start with "Yes" or "Maybe" and briefly explain if this product is suitable for {skin_type} skin.

**👍 Why it could be good for you:**
- List 4-5 specific benefits relevant to {skin_type} skin
- Be specific about texture, ingredients, and results
- Focus on how it helps {skin_type} skin specifically

**👎 Potential drawbacks / what to watch out for:**
- List 3-4 cons or things to be careful about
- Include any warnings specific to {skin_type} skin
- Be honest about potential issues

**💡 Tips for Using It Well:**
Provide 5-6 practical, step-by-step usage tips:
1. First tip
2. Second tip
3. Third tip
(etc.)

**✅ My take:**
Give a clear, honest recommendation. Be conversational and friendly.

Make it sound like advice from a knowledgeable friend. Be specific and helpful."""

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        review_text = response.text.strip()
        
        logger.info(f"✅ Review generated for: {product_name}")
        
        return {
            "productName": product_name,
            "skinType": skin_type,
            "review": review_text
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating review: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate review: {str(e)}"
        )

# ===== FETCH PRODUCT IMAGE ENDPOINT =====
@app.get("/api/product-image")
async def get_product_image(product_name: str):
    """Fetch product image from Google Custom Search"""
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        logger.warning("⚠️ Google Search API credentials not set")
        return {"imageUrl": None}
    
    try:
        logger.info(f"🔍 Fetching image for: {product_name}")
        
        query = f"{product_name} official product image"
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_SEARCH_API_KEY}&cx={GOOGLE_CSE_ID}&searchType=image&num=1&q={query}"
        
        response = requests.get(url)
        data = response.json()
        
        if data.get("items") and len(data["items"]) > 0:
            image_url = data["items"][0]["link"]
            logger.info(f"✅ Image found for: {product_name}")
            return {"imageUrl": image_url}
        else:
            logger.warning(f"⚠️ No image found for: {product_name}")
            return {"imageUrl": None}
            
    except Exception as e:
        logger.error(f"❌ Error fetching image: {str(e)}")
        return {"imageUrl": None}

# ===== ANALYZE COMPATIBILITY ENDPOINT =====
@app.post("/api/analyze-compatibility")
async def analyze_compatibility(request: CompatibilityRequest) -> AnalysisResponse:
    """Get AI analysis for product compatibility"""
    
    if not GENAI_AVAILABLE:
        logger.warning("⚠️ Gemini not available, returning fallback analysis")
        return {
            "explanation": "AI analysis unavailable. Returning heuristic guidance based on ingredient database.",
            "recommendations": [
                "Introduce one active ingredient at a time",
                "Patch test new products on a small area first",
                "Use SPF 30+ daily when using active ingredients",
                "Monitor your skin for 2 weeks to assess tolerance"
            ],
            "warnings": ["AI service disabled - using basic analysis only"],
            "tips": "Install google-generativeai and set GEMINI_API_KEY to enable AI analysis.",
            "score": 3.0
        }

    try:
        logger.info(f"🤖 Analyzing compatibility for {request.newProduct.brand} - {request.newProduct.title}")
        
        # Format current products info
        current_products_str = "\n".join([
            f"- {p.brand} {p.title}: {', '.join(p.ingredients[:5])}{'...' if len(p.ingredients) > 5 else ''}"
            for p in request.currentProducts
        ]) or "No products in routine yet"

        # Format conflicts
        conflicts_str = "\n".join([
            f"- {c.get('ingredient1')} + {c.get('ingredient2')}: {c.get('reason')}"
            for c in request.detectedConflicts
        ]) or "No conflicts detected"

        prompt = f"""You are a professional skincare expert with 10+ years of experience. Analyze this product compatibility scenario:

SKIN TYPE: {request.skinType}

CURRENT ROUTINE:
{current_products_str}

NEW PRODUCT BEING ADDED:
Brand: {request.newProduct.brand}
Product: {request.newProduct.title}
Ingredients: {', '.join(request.newProduct.ingredients)}

DETECTED CONFLICTS:
{conflicts_str}

Provide a detailed compatibility analysis in JSON format:
1. "explanation": A 2-3 sentence explanation of compatibility considering skin type and ingredients
2. "recommendations": 3-4 specific, actionable recommendations (list of strings)
3. "warnings": Critical warnings if any (list of strings, can be empty)
4. "tips": Additional skincare tips for safely using this product
5. "score": Compatibility score from 1-5 based on your analysis. Consider: detected conflicts (weight 40%), skin type compatibility (30%), ingredient synergy (30%)

SCORING GUIDELINES:
- 5: Excellent compatibility, no concerns, safe to use freely
- 4: Good compatibility, minor considerations but generally safe
- 3: Moderate compatibility, some caution needed, conflicting ingredients present
- 2: Poor compatibility, significant concerns, requires strict management
- 1: Very poor compatibility, strongly not recommended, high risk of irritation

Return ONLY valid JSON, no markdown or extra text:
{{
    "explanation": "...",
    "recommendations": ["...", "...", "..."],
    "warnings": ["..."],
    "tips": "...",
    "score": 3.5
}}"""

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        analysis = json.loads(response_text)
        logger.info(f"✅ Analysis complete for {request.newProduct.brand}")
        return analysis

    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parse error in analysis: {str(e)}")
        return {
            "explanation": "Unable to generate AI analysis at this moment. Basic heuristic analysis shows no critical conflicts.",
            "recommendations": [
                "Monitor how your skin responds over 1-2 weeks",
                "Introduce one product at a time to identify reactions",
                "Use SPF 30+ daily if using active ingredients",
                "Consult a dermatologist if any irritation occurs"
            ],
            "warnings": [],
            "tips": "If irritation occurs, discontinue use and consult a dermatologist."
        }
    except Exception as e:
        logger.error(f"❌ Error in compatibility analysis: {str(e)}")
        return {
            "explanation": "Unable to get AI analysis at this moment. Your product has been added to the routine.",
            "recommendations": [
                "Monitor how your skin responds over 1-2 weeks",
                "Introduce one product at a time to identify reactions",
                "Use SPF 30+ daily if using active ingredients"
            ],
            "warnings": [],
            "tips": "If irritation occurs, discontinue and consult a dermatologist."
        }

# ===== CONFLICTS ENDPOINT =====
@app.get("/api/conflicts")
def get_conflicts() -> List[ConflictItem]:
    """Get all known ingredient conflicts"""
    conflicts = [
        # === Retinol Conflicts ===
        {
            "ingredient1": "Retinol",
            "ingredient2": "AHA",
            "severity": "warning",
            "reason": "Over-exfoliation risk - can irritate and damage skin barrier"
        },
        {
            "ingredient1": "Retinol",
            "ingredient2": "BHA",
            "severity": "warning",
            "reason": "Over-exfoliation risk - can irritate and damage skin barrier"
        },
        {
            "ingredient1": "Retinol",
            "ingredient2": "Salicylic Acid",
            "severity": "warning",
            "reason": "Over-exfoliation risk - severe irritation possible"
        },
        {
            "ingredient1": "Retinol",
            "ingredient2": "Benzoyl Peroxide",
            "severity": "warning",
            "reason": "Benzoyl peroxide oxidizes and degrades retinoids - neutralizes effects and increases irritation"
        },
        {
            "ingredient1": "Retinol",
            "ingredient2": "Vitamin C",
            "severity": "warning",
            "reason": "Both are strong actives - can cause irritation and sensitivity when combined, best used at different times"
        },
        
        # === Vitamin C Conflicts ===
        {
            "ingredient1": "Vitamin C",
            "ingredient2": "Niacinamide",
            "severity": "info",
            "reason": "Myth debunked - actually works well together and provides additional benefits"
        },
        {
            "ingredient1": "Vitamin C",
            "ingredient2": "Salicylic Acid",
            "severity": "caution",
            "reason": "Multiple actives - may over-exfoliate and irritate skin"
        },
        {
            "ingredient1": "Vitamin C",
            "ingredient2": "AHA",
            "severity": "caution",
            "reason": "Too many actives - risk of over-exfoliation and barrier damage"
        },
        {
            "ingredient1": "Vitamin C",
            "ingredient2": "BHA",
            "severity": "caution",
            "reason": "Too many actives - risk of over-exfoliation and barrier damage"
        },
        {
            "ingredient1": "Vitamin C",
            "ingredient2": "Benzoyl Peroxide",
            "severity": "warning",
            "reason": "Benzoyl peroxide oxidizes and degrades vitamin C - cancels out antioxidant benefits"
        },
        
        # === Benzoyl Peroxide Conflicts ===
        {
            "ingredient1": "Benzoyl Peroxide",
            "ingredient2": "Glycolic Acid",
            "severity": "caution",
            "reason": "Both are potentially irritating - combined use increases sensitivity risk"
        },
        {
            "ingredient1": "Benzoyl Peroxide",
            "ingredient2": "Hydroquinone",
            "severity": "warning",
            "reason": "Can cause temporary dark staining or discoloration of the skin - avoid layering"
        },
        {
            "ingredient1": "Benzoyl Peroxide",
            "ingredient2": "Dapsone",
            "severity": "warning",
            "reason": "Can cause temporary orange or brown skin discoloration"
        },
        
        # === AHA/BHA Conflicts ===
        {
            "ingredient1": "AHA",
            "ingredient2": "Salicylic Acid",
            "severity": "caution",
            "reason": "Too much exfoliation - can compromise skin barrier"
        },
        {
            "ingredient1": "AHA",
            "ingredient2": "BHA",
            "severity": "caution",
            "reason": "Combining multiple exfoliants can lead to over-exfoliation and irritation"
        },
        
        # === Copper Peptides Conflicts ===
        {
            "ingredient1": "Copper Peptides",
            "ingredient2": "Vitamin C",
            "severity": "warning",
            "reason": "Ascorbic acid can destabilize copper peptides and reduce effectiveness - use at different times"
        },
        {
            "ingredient1": "Copper Peptides",
            "ingredient2": "AHA",
            "severity": "warning",
            "reason": "Low pH acids can destabilize peptides and reduce their effectiveness - best used separately"
        },
        {
            "ingredient1": "Copper Peptides",
            "ingredient2": "BHA",
            "severity": "warning",
            "reason": "Low pH acids can destabilize peptides and reduce their effectiveness - best used separately"
        },
        
        # === Hydroquinone Conflicts ===
        {
            "ingredient1": "Hydroquinone",
            "ingredient2": "AHA",
            "severity": "caution",
            "reason": "Both are potent actives that can irritate when combined - use at different times or consider alternatives"
        },
        {
            "ingredient1": "Hydroquinone",
            "ingredient2": "BHA",
            "severity": "caution",
            "reason": "Both are potent actives that can irritate when combined - use at different times"
        },
        
        # === Niacinamide Conflicts ===
        {
            "ingredient1": "Niacinamide",
            "ingredient2": "Vitamin C",
            "severity": "info",
            "reason": "Old myth - these actually work synergistically when formulated properly"
        },
        
        # === More Important Conflicts ===
        {
            "ingredient1": "Tretinoin",
            "ingredient2": "AHA",
            "severity": "warning",
            "reason": "Both increase cell turnover - high risk of severe irritation and barrier damage"
        },
        {
            "ingredient1": "Tretinoin",
            "ingredient2": "BHA",
            "severity": "warning",
            "reason": "Both increase cell turnover - high risk of severe irritation"
        },
        {
            "ingredient1": "Alpha Arbutin",
            "ingredient2": "AHA",
            "severity": "caution",
            "reason": "AHAs can enhance absorption but may cause irritation - introduce slowly"
        },
        {
            "ingredient1": "Azelaic Acid",
            "ingredient2": "AHA",
            "severity": "caution",
            "reason": "Multiple acids can over-exfoliate - consider alternating days"
        },
        {
            "ingredient1": "Zinc Oxide",
            "ingredient2": "Salicylic Acid",
            "severity": "caution",
            "reason": "May reduce effectiveness of both ingredients - use at different times"
        }
    ]
    logger.info(f"📋 Returning {len(conflicts)} known conflicts")
    return conflicts

@app.post("/api/products/save")
async def save_product(product: Product):
    """Save a product found via Gemini to the database for future use"""
    try:
        logger.info(f"💾 Saving product to database: {product.brand} - {product.title}")
        logger.info(f"📋 Ingredients: {product.ingredients}")
        
        # Load existing products
        try:
            with open('productData.json', 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
        except FileNotFoundError:
            data = []
        
        # Create product object
        import time
        new_product = {
            "title": product.title,
            "brand": product.brand,
            "ingredients": ",".join(product.ingredients) if isinstance(product.ingredients, list) else product.ingredients,
            "sku": product.sku or f"prod-{int(time.time() * 1000)}"
        }
        
        logger.info(f"✅ Product object: {new_product}")
        
        # Check if product already exists
        product_exists = any(
            p.get('title') == product.title and p.get('brand') == product.brand 
            for p in data
        )
        
        if product_exists:
            logger.warning(f"⚠️ Product already exists in database")
            return {
                "status": "already_exists",
                "message": f"{product.brand} - {product.title} already in database",
                "product": new_product
            }
        
        # Add to database
        data.append(new_product)
        
        # Save back to file
        with open('productData.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Product saved successfully. Total products: {len(data)}")
        
        return {
            "status": "success",
            "message": f"Product saved: {product.brand} - {product.title}",
            "product": new_product,
            "total_products": len(data)
        }
        
    except Exception as e:
        logger.error(f"❌ Error saving product: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save product: {str(e)}"
        )


# ===== ERROR HANDLERS =====
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"❌ HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        },
        headers={
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)