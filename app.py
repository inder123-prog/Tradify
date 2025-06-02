import os
import requests
import logging
from datetime import datetime, timedelta
import threading
from flask import Flask, render_template, jsonify, request
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# --- Configuration ---
# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Finnhub API token from environment variable for security
FINNHUB_TOKEN = "d0v0ir9r01qmg3ujgiv0d0v0ir9r01qmg3ujgivg"
if not FINNHUB_TOKEN:
    raise ValueError("FINNHUB_API_TOKEN environment variable not set. Please set it before running.")

# --- Sentiment Analysis Configuration ---
SENTIMENT_KEYWORDS = {
    'positive': ['good', 'great', 'up', 'high', 'positive', 'strong', 'bullish', 'beat', 'outperform', 'upgrade', 'rally', 'gain', 'profit', 'optimistic', 'record'],
    'negative': ['bad', 'not good', 'down', 'low', 'negative', 'weak', 'bearish', 'miss', 'underperform', 'downgrade', 'slump', 'loss', 'pessimistic', 'plunge', 'drop', 'slashed'],
    'uncertainty_fear': ['fear', 'uncertain', 'risk', 'volatile', 'volatility', 'warning', 'concern', 'jittery', 'turmoil', 'headwinds', 'investigation', 'lawsuit', 'cut']
}

GLOBAL_NEWS_KEYWORDS = {
    'geopolitical': ['war', 'conflict', 'invasion', 'sanctions', 'tensions', 'geopolitical'],
    'macroeconomic': ['inflation', 'interest rates', 'recession', 'federal reserve', 'fed', 'gdp', 'unemployment'],
    'health_crisis': ['pandemic', 'outbreak', 'virus', 'lockdown'],
    'trade': ['trade war', 'tariffs', 'supply chain']
}

# --- In-memory storage for analysis results (Thread-safe) ---
# In a production system, you'd use a database (like Redis, PostgreSQL) instead.
analysis_results = {}
results_lock = threading.Lock()

# --- Core Logic Functions ---
def perform_sentiment_analysis(text):
    """
    Performs a simple keyword-based sentiment analysis on a given text.
    Returns a dictionary with scores.
    """
    scores = {'positive': 0, 'negative': 0, 'uncertainty_fear': 0}
    words_found = {'positive': [], 'negative': [], 'uncertainty_fear': []}
    
    # Simple tokenization by splitting on spaces and punctuation
    # A more robust solution would use NLTK or spaCy
    text_lower = text.lower()
    words = text_lower.split() 

    for category, keywords in SENTIMENT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower: # Check substring for multi-word keywords
                scores[category] += 1
                if keyword not in words_found[category]:
                    words_found[category].append(keyword)
    
    return scores, words_found

def analyze_global_impact(text):
    """
    Placeholder to check for global news keywords.
    Returns a list of detected categories.
    """
    impact_categories = []
    text_lower = text.lower()
    for category, keywords in GLOBAL_NEWS_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower and category not in impact_categories:
                impact_categories.append(category)
    return impact_categories

def fetch_and_analyze_news(ticker_symbol):
    """
    The main function that is scheduled to run.
    It fetches news, performs analysis, and stores the results.
    """
    logging.info(f"SCHEDULER: Starting news analysis for {ticker_symbol}...")
    
    # Set a rolling 24-hour window for news fetching
    to_date = datetime.now()
    from_date = to_date - timedelta(days=1)
    from_date_str = from_date.strftime('%Y-%m-%d')
    to_date_str = to_date.strftime('%Y-%m-%d')
    
    api_url = f"https://finnhub.io/api/v1/company-news?symbol={ticker_symbol}&from={from_date_str}&to={to_date_str}&token={FINNHUB_TOKEN}"
    
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        news_items = response.json()
        
        if not news_items:
            logging.info(f"SCHEDULER: No news found for {ticker_symbol} in the last 24 hours.")
            return

        total_scores = {'positive': 0, 'negative': 0, 'uncertainty_fear': 0}
        analyzed_articles = []
        
        for item in news_items:
            # Combine headline and summary for a more comprehensive analysis
            text_to_analyze = f"{item.get('headline', '')}. {item.get('summary', '')}"
            
            sentiment_scores, sentiment_words = perform_sentiment_analysis(text_to_analyze)
            global_impacts = analyze_global_impact(text_to_analyze)

            total_scores['positive'] += sentiment_scores['positive']
            total_scores['negative'] += sentiment_scores['negative']
            total_scores['uncertainty_fear'] += sentiment_scores['uncertainty_fear']
            
            analyzed_articles.append({
                "headline": item.get('headline'),
                "summary": item.get('summary'),
                "url": item.get('url'),
                "source": item.get('source'),
                "publish_time": datetime.fromtimestamp(item.get('datetime')).strftime('%Y-%m-%d %H:%M:%S'),
                "sentiment_scores": sentiment_scores,
                "sentiment_words_found": sentiment_words,
                "global_impact_categories": global_impacts
            })

        # Determine overall sentiment
        final_score = total_scores['positive'] - total_scores['negative'] - (total_scores['uncertainty_fear'] * 0.5) # Penalize uncertainty
        overall_sentiment = "Neutral"
        if final_score > 2: overall_sentiment = "Positive"
        elif final_score < -2: overall_sentiment = "Negative"
        elif total_scores['uncertainty_fear'] > total_scores['positive'] + total_scores['negative']:
            overall_sentiment = "Uncertain"
        
        # Store results safely using the lock
        with results_lock:
            analysis_results[ticker_symbol] = {
                "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ticker": ticker_symbol,
                "overall_sentiment": overall_sentiment,
                "final_score": final_score,
                "aggregated_scores": total_scores,
                "article_count": len(analyzed_articles),
                "analyzed_articles": analyzed_articles
            }
        logging.info(f"SCHEDULER: Finished analysis for {ticker_symbol}. Overall Sentiment: {overall_sentiment}")

    except requests.exceptions.RequestException as e:
        logging.error(f"SCHEDULER: API request failed for {ticker_symbol}. Error: {e}")
    except Exception as e:
        logging.error(f"SCHEDULER: An unexpected error occurred during analysis for {ticker_symbol}. Error: {e}", exc_info=True)


# --- Flask Application ---
app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/start-analysis', methods=['POST'])
def start_analysis():
    """API endpoint to start the scheduled analysis for a given ticker."""
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    if not ticker:
        return jsonify({"error": "Ticker symbol is required."}), 400

    job_id = f"job_{ticker}"
    
    # Check if a job for this ticker already exists
    if scheduler.get_job(job_id):
        logging.info(f"Job for {ticker} already exists. Triggering a run now.")
        scheduler.get_job(job_id).modify(next_run_time=datetime.now()) # Run it now
    else:
        logging.info(f"Adding new hourly job for {ticker}.")
        scheduler.add_job(
            fetch_and_analyze_news,
            trigger='interval',
            hours=1,
            id=job_id,
            args=[ticker],
            replace_existing=True,
            next_run_time=datetime.now() # Run immediately for the first time
        )
    
    return jsonify({"message": f"Hourly news analysis scheduled for {ticker}. First analysis is running now."}), 202

@app.route('/get-results/<ticker>')
def get_results(ticker):
    """API endpoint to fetch the latest analysis results."""
    ticker = ticker.upper()
    with results_lock:
        result = analysis_results.get(ticker)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"message": "No analysis results found yet. Please start analysis or wait for the first run to complete."}), 404

# --- Scheduler Setup ---
scheduler = BackgroundScheduler(daemon=True)
# Register the shutdown function to be called on exit
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    # Start the scheduler when the app runs
    scheduler.start()
    logging.info("Background scheduler started.")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False) # use_reloader=False is important for APScheduler
