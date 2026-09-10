import requests
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class MacroEconomicFetcher:
    """
    Fetches macroeconomic data relevant for debt decisions for a given country.
    Uses the World Bank API (no API key required).
    """
    
    # World Bank Indicator IDs
    INDICATORS = {
        "inflation_rate": "FP.CPI.TOTL.ZG",       # Inflation, consumer prices (annual %)
        "unemployment_rate": "SL.UEM.TOTL.ZS",    # Unemployment, total (% of total labor force)
        "real_interest_rate": "FR.INR.RINR",      # Real interest rate (%)
        "gdp_growth": "NY.GDP.MKTP.KD.ZG"         # GDP growth (annual %)
    }
    
    def __init__(self):
        self.base_url = "https://api.worldbank.org/v2/country"
        
    def _fetch_indicator(self, country_code: str, indicator: str) -> float:
        """Fetch latest non-null value for an indicator."""
        
        url = f"{self.base_url}/{country_code}/indicator/{indicator}"

        params = {
            "format": "json",
            "per_page": 100
        }

        try:
            response = requests.get(url, params=params, timeout=10)

            print("REQUEST URL:", response.url)   # debug

            response.raise_for_status()

            data = response.json()

            # Debug API response
            # print(data)

            if len(data) > 1:

                # Find first non-null value
                for entry in data[1]:
                    value = entry.get("value")

                    if value is not None:
                        return float(value)

            return 0.0

        except Exception as e:
            logger.warning(
                f"Failed to fetch {indicator} for {country_code}: {e}"
            )
            return 0.0

    def get_macro_context(self, country_code: str) -> Dict[str, float]:
        """
        Gathers macroeconomic factors for the specified country.
        
        Args:
            country_code (str): The 2-letter or 3-letter ISO country code (e.g., 'us', 'gb', 'gr').
            
        Returns:
            Dict[str, float]: A dictionary mapping the indicator name to its numeric value.
        """
        macro_values = {}
        for friendly_name, indicator_id in self.INDICATORS.items():
            value = self._fetch_indicator(country_code, indicator_id)
            macro_values[friendly_name] = round(value, 2)
            
        return macro_values

# Example usage
if __name__ == "__main__":
    fetcher = MacroEconomicFetcher()
    # Fetch for United States
    print("US Data:", fetcher.get_macro_context("us"))
    # Fetch for Greece
    print("GR Data:", fetcher.get_macro_context("gr"))
