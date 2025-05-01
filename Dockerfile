# Start met Python als basis
FROM python:3.9-slim

# Installeer Node.js
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    && curl -sL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Installeer Chrome en benodigde dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    default-jdk \
    libglib2.0-0 \
    libnss3 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxss1 \
    libasound2 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libfontconfig1 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    python3-tk \
    # Tesseract en afhankelijkheden voor OCR
    tesseract-ocr \
    libtesseract-dev \
    tesseract-ocr-eng \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Gebruik Chromium in plaats van Chrome (werkt op ARM en x86)
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Installeer de nieuwste beschikbare ChromeDriver versie (voor Chrome 123)
# Chrome 134 is te nieuw, dus we gebruiken de laatste beschikbare versie
RUN echo "Using chromedriver from Chromium package" \
    && ln -sf /usr/bin/chromedriver /usr/local/bin/chromedriver

# Set up Chrome environment variables
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV DISPLAY=:99

# Installeer Playwright-specifieke dependencies
# Bijgewerkte versies voor Debian Bookworm
RUN apt-get update && apt-get install -y \
    fonts-noto-color-emoji \
    libopus0 \
    libwebp7 \
    libenchant-2-2 \
    libgudev-1.0-0 \
    libsecret-1-0 \
    libhyphen0 \
    libvpx7 \
    libevent-2.1-7 \
    ffmpeg \
    libwoff1 \
    libharfbuzz-icu0 \
    && rm -rf /var/lib/apt/lists/*

# Werkdirectory instellen
WORKDIR /app

# Kopieer alleen requirements.txt
COPY requirements.txt .

# Installeer benodigde systeem packages voor Python
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Installeer yfinance expliciet
RUN pip install yfinance==0.2.57

# Installeer cachetools expliciet (nodig voor YahooFinanceProvider)
RUN pip install cachetools>=5.5.0

# Installeer alle Python dependencies
RUN pip install -r requirements.txt

# Installeer Playwright voor Python en Node.js
RUN pip install playwright && playwright install chromium
RUN npm install playwright@latest && npx playwright install chromium

# Kopieer de rest van de app
COPY . .

# Poort voor FastAPI
EXPOSE 8000

# Start de applicatie
CMD ["python", "-m", "trading_bot.main"]
