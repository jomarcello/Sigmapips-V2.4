"""
API Connector module met ingebouwd retry mechanisme.
Dit script biedt een robuuste manier om met externe API's te verbinden.
"""

import time
import random
import logging
import requests
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

# Type definition for the return value of a function
T = TypeVar('T')

class APIConnector:
    """
    Een klasse die API-verbindingen beheert met ingebouwde retry functionaliteit.
    Gebruikt exponentiële backoff voor een betrouwbaardere verbinding.
    """
    
    def __init__(
        self, 
        base_url: str,
        max_retries: int = 5,
        backoff_factor: float = 0.5,
        jitter: bool = True,
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialiseert de API connector.
        
        Args:
            base_url: Basis URL voor de API
            max_retries: Maximum aantal retries voordat opgegeven wordt
            backoff_factor: Factor voor exponentiële backoff calculatie
            jitter: Of er een willekeurige variatie moet worden toegevoegd aan de wachttijden
            timeout: Timeout in seconden voor elke request
            headers: Standaard headers voor alle requests
            logger: Logger object om berichten naar te loggen
        """
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.timeout = timeout
        self.headers = headers or {}
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        
        # Standaard headers instellen
        self.session.headers.update(self.headers)
    
    def with_retry(
        self, 
        func: Callable[..., T], 
        *args: Any, 
        retries: Optional[int] = None, 
        **kwargs: Any
    ) -> T:
        """
        Voert een functie uit met automatische retry bij falen.
        
        Args:
            func: De functie die uitgevoerd moet worden
            *args: Argumenten voor de functie
            retries: Optioneel aantal retries, gebruikt max_retries als None
            **kwargs: Keyword argumenten voor de functie
            
        Returns:
            Het resultaat van de functie
            
        Raises:
            Exception: Doorgestuurd van de uitgevoerde functie na alle retries
        """
        retries = self.max_retries if retries is None else retries
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:  # Vang alle exceptions
                last_exception = e
                if attempt == retries:
                    # Als dit de laatste poging was, geef de fout door
                    self.logger.error(f"Maximum retries ({retries}) reached. Last error: {str(e)}")
                    raise
                
                # Bereken wachttijd met exponentiële backoff
                wait_time = self._calculate_backoff(attempt)
                
                self.logger.warning(
                    f"Error on attempt {attempt + 1}/{retries + 1}. "
                    f"Retrying in {wait_time:.2f} seconds. Error: {str(e)}"
                )
                
                # Wacht voordat de volgende poging wordt gedaan
                time.sleep(wait_time)
    
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Berekent de wachttijd voor de volgende poging met exponentiële backoff.
        
        Args:
            attempt: Het huidige aantal pogingen (0-based)
            
        Returns:
            Wachttijd in seconden
        """
        wait_time = self.backoff_factor * (2 ** attempt)
        
        if self.jitter:
            # Voeg een willekeurige variatie toe tussen 0% en 25% van de wachttijd
            jitter_amount = random.uniform(0, 0.25 * wait_time)
            wait_time += jitter_amount
            
        return wait_time
    
    def get(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None
    ) -> requests.Response:
        """
        Voert een GET request uit met automatische retry.
        
        Args:
            endpoint: API endpoint om aan te roepen
            params: Query parameters voor de request
            headers: Extra headers voor deze specifieke request
            retries: Optioneel aantal retries, gebruikt max_retries als None
            
        Returns:
            Response object van de request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self.headers.copy()
        
        if headers:
            request_headers.update(headers)
            
        self.logger.debug(f"Making GET request to {url}")
            
        def _make_request() -> requests.Response:
            response = self.session.get(
                url, 
                params=params, 
                headers=request_headers, 
                timeout=self.timeout
            )
            response.raise_for_status()  # Raise exception voor 4XX/5XX statuscodes
            return response
            
        return self.with_retry(_make_request, retries=retries)
    
    def post(
        self, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None, 
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None
    ) -> requests.Response:
        """
        Voert een POST request uit met automatische retry.
        
        Args:
            endpoint: API endpoint om aan te roepen
            data: Form data voor de request
            json_data: JSON data voor de request
            params: Query parameters voor de request
            headers: Extra headers voor deze specifieke request
            retries: Optioneel aantal retries, gebruikt max_retries als None
            
        Returns:
            Response object van de request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self.headers.copy()
        
        if headers:
            request_headers.update(headers)
            
        self.logger.debug(f"Making POST request to {url}")
            
        def _make_request() -> requests.Response:
            response = self.session.post(
                url, 
                data=data,
                json=json_data,
                params=params, 
                headers=request_headers, 
                timeout=self.timeout
            )
            response.raise_for_status()  # Raise exception voor 4XX/5XX statuscodes
            return response
            
        return self.with_retry(_make_request, retries=retries)
    
    def put(
        self, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None, 
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None
    ) -> requests.Response:
        """
        Voert een PUT request uit met automatische retry.
        
        Args:
            endpoint: API endpoint om aan te roepen
            data: Form data voor de request
            json_data: JSON data voor de request
            params: Query parameters voor de request
            headers: Extra headers voor deze specifieke request
            retries: Optioneel aantal retries, gebruikt max_retries als None
            
        Returns:
            Response object van de request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self.headers.copy()
        
        if headers:
            request_headers.update(headers)
            
        self.logger.debug(f"Making PUT request to {url}")
            
        def _make_request() -> requests.Response:
            response = self.session.put(
                url, 
                data=data,
                json=json_data,
                params=params, 
                headers=request_headers, 
                timeout=self.timeout
            )
            response.raise_for_status()  # Raise exception voor 4XX/5XX statuscodes
            return response
            
        return self.with_retry(_make_request, retries=retries)
    
    def delete(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        headers: Optional[Dict[str, str]] = None,
        retries: Optional[int] = None
    ) -> requests.Response:
        """
        Voert een DELETE request uit met automatische retry.
        
        Args:
            endpoint: API endpoint om aan te roepen
            params: Query parameters voor de request
            headers: Extra headers voor deze specifieke request
            retries: Optioneel aantal retries, gebruikt max_retries als None
            
        Returns:
            Response object van de request
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        request_headers = self.headers.copy()
        
        if headers:
            request_headers.update(headers)
            
        self.logger.debug(f"Making DELETE request to {url}")
            
        def _make_request() -> requests.Response:
            response = self.session.delete(
                url, 
                params=params, 
                headers=request_headers, 
                timeout=self.timeout
            )
            response.raise_for_status()  # Raise exception voor 4XX/5XX statuscodes
            return response
            
        return self.with_retry(_make_request, retries=retries)
    
    def close(self) -> None:
        """Sluit de sessie."""
        self.session.close()
    
    def __enter__(self) -> 'APIConnector':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit, sluit de sessie."""
        self.close() 