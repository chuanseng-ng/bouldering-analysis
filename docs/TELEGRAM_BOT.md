# Telegram Bot Frontend Guide

**Last Updated**: 2026-01-30
**Purpose**: Setup and implementation guide for the Telegram bot frontend

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Bot Setup](#bot-setup)
4. [Local Development](#local-development)
5. [Implementation](#implementation)
6. [Deployment](#deployment)
7. [User Guide](#user-guide)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Telegram bot provides a **lightweight, mobile-friendly alternative** to the web frontend for quick route analysis.

### Use Cases

- **Gym Climbers**: Get quick grade estimates while at the gym
- **Mobile Users**: No need to open a browser or install an app
- **Quick Checks**: Minimal friction for simple analysis
- **Sharing**: Easy to share bot with friends

### Features

**Current (MVP)**:
- ✅ Photo upload via Telegram chat
- ✅ Automatic grade prediction
- ✅ Text-based explanations
- ✅ Simple commands (`/start`, `/help`)

**Future Enhancements**:
- Route history with thumbnails
- Feedback submission via inline buttons
- Hold annotation (advanced users)
- Multi-language support

### Architecture

```text
User → Telegram App → Telegram API → Bot Server → FastAPI Backend → Supabase
```

**Bot Modes**:
- **Polling** (Development): Bot actively checks for new messages
- **Webhook** (Production): Telegram pushes updates to your server

---

## Prerequisites

### Required

- [ ] Python 3.10+
- [ ] Telegram account
- [ ] Backend API running (local or deployed)
- [ ] Supabase configured (for image storage)

### Optional

- [ ] Server with public IP (for webhook mode)
- [ ] Domain name with SSL certificate (for production webhook)
- [ ] Cloud platform account (AWS, GCP, Azure) for serverless deployment

---

## Bot Setup

### Step 1: Create Telegram Bot

1. **Open Telegram** and search for `@BotFather`

2. **Start Chat** with BotFather and send `/newbot`

3. **Choose Bot Name**:
   ```
   BotFather: Alright, a new bot. How are we going to call it?
   You: Bouldering Route Analyzer
   ```

4. **Choose Username** (must end with `bot`):
   ```
   BotFather: Good. Now let's choose a username for your bot.
   You: bouldering_analysis_bot
   ```

5. **Save the Token**:
   ```
   BotFather: Done! Your token is: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   ```

   **Important**: Keep this token secret! Anyone with this token can control your bot.

### Step 2: Configure Bot Settings (Optional)

1. **Set Description** (`/setdescription`):
   ```
   Analyze bouldering routes and get grade predictions instantly!
   Just send me a photo of a climbing route.
   ```

2. **Set About Text** (`/setabouttext`):
   ```
   AI-powered bouldering route analysis bot
   ```

3. **Set Bot Picture** (`/setuserpic`):
   - Upload a climbing-themed image (512x512 recommended)

4. **Set Commands** (`/setcommands`):
   ```
   start - Start the bot and see welcome message
   help - Show usage instructions
   analyze - Analyze a route photo
   history - View your recent analyses (coming soon)
   feedback - Submit feedback on a prediction
   ```

### Step 3: Environment Configuration

Create a `.env` file for the bot:

```bash
# .env.telegram-bot
# IMPORTANT: Add .env.telegram-bot to .gitignore and use placeholder values here.

# Telegram Configuration
TELEGRAM_BOT_TOKEN=YOUR_TELEGRAM_BOT_TOKEN
TELEGRAM_WEBHOOK_URL=https://yourdomain.com/webhook  # For production

# Backend API
TELEGRAM_API_URL=http://localhost:8000  # Or production URL

# Optional
TELEGRAM_ADMIN_USER_IDS=123456789,987654321  # Admin Telegram user IDs
TELEGRAM_MAX_FILE_SIZE_MB=10
```

---

## Local Development

### Step 1: Install Dependencies

Create a separate directory for the bot (or add to main project):

```bash
# Option 1: Separate directory
mkdir telegram-bot
cd telegram-bot

# Option 2: Add to main project
cd bouldering-analysis
mkdir -p src/telegram_bot
```

Create `requirements.txt`:

```text
python-telegram-bot==20.7
python-dotenv==1.0.0
httpx==0.28.1
pillow==10.4.0
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Step 2: Create Bot Application

Create `bot.py`:

```python
"""Telegram bot for bouldering route analysis."""

import logging
import os
from io import BytesIO
from typing import Any

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = os.getenv("TELEGRAM_API_URL", "http://localhost:8000")

# Logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    welcome_message = (
        "🧗 Welcome to Bouldering Route Analyzer!\n\n"
        "Send me a photo of a climbing route and I'll estimate its grade.\n\n"
        "Commands:\n"
        "/help - Show usage instructions\n"
        "/analyze - Analyze a route (send photo after this command)\n\n"
        "Just send me a photo to get started!"
    )
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_message = (
        "📖 How to use this bot:\n\n"
        "1️⃣ Take a photo of a bouldering route\n"
        "2️⃣ Send it to me\n"
        "3️⃣ Wait for analysis (takes ~5-10 seconds)\n"
        "4️⃣ Get your grade prediction!\n\n"
        "Tips:\n"
        "• Include the full route in the photo\n"
        "• Make sure holds are visible\n"
        "• Good lighting helps accuracy\n\n"
        "Questions? Contact @your_username"
    )
    await update.message.reply_text(help_message)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads and analyze routes."""
    user = update.effective_user
    logger.info(f"Photo received from {user.username} ({user.id})")

    # Send "analyzing" message
    status_message = await update.message.reply_text(
        "🔍 Analyzing your route...\nThis may take a few seconds."
    )

    try:
        # Get the largest photo size
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)

        # Download photo
        photo_bytes = BytesIO()
        await file.download_to_memory(photo_bytes)
        photo_bytes.seek(0)

        # Upload to backend
        async with httpx.AsyncClient() as client:
            # Step 1: Upload image
            files = {"file": ("route.jpg", photo_bytes, "image/jpeg")}
            upload_response = await client.post(
                f"{API_URL}/api/v1/routes/upload",
                files=files,
                timeout=30.0,
            )
            upload_response.raise_for_status()
            upload_data = upload_response.json()

            # Step 2: Create route record
            route_response = await client.post(
                f"{API_URL}/api/v1/routes",
                json={"image_url": upload_data["public_url"]},
                timeout=10.0,
            )
            route_response.raise_for_status()
            route_data = route_response.json()
            route_id = route_data["id"]

            # Step 3: Trigger analysis (if endpoint exists)
            # This would be implemented when backend analysis is ready
            # analyze_response = await client.post(
            #     f"{API_URL}/api/v1/routes/{route_id}/analyze",
            #     timeout=60.0,
            # )
            # analyze_response.raise_for_status()

            # Step 4: Get prediction (placeholder for now)
            # prediction_response = await client.get(
            #     f"{API_URL}/api/v1/routes/{route_id}/prediction",
            #     timeout=10.0,
            # )
            # prediction_response.raise_for_status()
            # prediction_data = prediction_response.json()

            # For now, send a placeholder response
            result_message = (
                "✅ Route uploaded successfully!\n\n"
                f"🆔 Route ID: {route_id}\n"
                f"📸 Image stored at: {upload_data['public_url'][:50]}...\n\n"
                "⚠️ Note: Full analysis pipeline not yet implemented.\n"
                "Once complete, you'll receive:\n"
                "• Grade prediction (V-scale)\n"
                "• Confidence level\n"
                "• Explanation of key factors\n"
            )

            await status_message.edit_text(result_message)
            logger.info(f"Successfully processed route {route_id}")

    except httpx.HTTPError as e:
        logger.error(f"API error: {e}")
        await status_message.edit_text(
            "❌ Sorry, there was an error connecting to the analysis service.\n"
            "Please try again later."
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await status_message.edit_text(
            "❌ Sorry, something went wrong.\n"
            "Please try again or contact support."
        )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages."""
    await update.message.reply_text(
        "Please send me a photo of a climbing route to analyze.\n"
        "Use /help for more information."
    )


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")


def main() -> None:
    """Start the bot."""
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    application.add_error_handler(error_handler)

    # Start bot (polling mode for development)
    logger.info("Starting bot in polling mode...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
```

### Step 3: Run Bot Locally

```bash
# Make sure backend is running
# In terminal 1:
cd /path/to/bouldering-analysis
uvicorn src.app:application --reload

# In terminal 2:
cd /path/to/telegram-bot
python bot.py
```

### Step 4: Test Bot

1. Open Telegram on your phone
2. Search for your bot (`@bouldering_analysis_bot`)
3. Send `/start`
4. Send a photo of a climbing route
5. Verify upload works

---

## Implementation

### Enhanced Features

Once you have the basic bot working, add these features:

#### 1. Full Analysis Integration

Update `handle_photo` to include full analysis pipeline:

```python
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photo uploads and analyze routes."""
    # ... (upload code from above)

    try:
        # Upload image
        upload_data = await upload_image(photo_bytes)
        route_id = upload_data["route_id"]

        # Trigger analysis
        await status_message.edit_text("🔍 Detecting holds...")
        await trigger_analysis(route_id)

        # Get prediction
        await status_message.edit_text("🤔 Calculating grade...")
        prediction = await get_prediction(route_id)

        # Format result
        result_message = format_prediction_message(prediction)

        await status_message.edit_text(result_message)

    except Exception as e:
        # Handle errors
        ...
```

#### 2. Inline Buttons for Feedback

Add interactive buttons:

```python
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

async def send_prediction_with_feedback(
    update: Update,
    route_id: str,
    prediction: dict
) -> None:
    """Send prediction with feedback buttons."""
    message = format_prediction_message(prediction)

    # Create inline keyboard
    keyboard = [
        [
            InlineKeyboardButton("👍 Accurate", callback_data=f"accurate:{route_id}"),
            InlineKeyboardButton("👎 Not Accurate", callback_data=f"inaccurate:{route_id}"),
        ],
        [
            InlineKeyboardButton("📝 Add Comment", callback_data=f"comment:{route_id}"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(message, reply_markup=reply_markup)


# Add callback handler
from telegram.ext import CallbackQueryHandler

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()

    action, route_id = query.data.split(":")

    if action == "accurate":
        # Submit positive feedback
        await submit_feedback(route_id, is_accurate=True)
        await query.edit_message_text("✅ Thanks for your feedback!")

    elif action == "inaccurate":
        # Submit negative feedback
        await submit_feedback(route_id, is_accurate=False)
        await query.edit_message_text("✅ Thanks! Your feedback helps improve the model.")

# In main():
application.add_handler(CallbackQueryHandler(button_callback))
```

#### 3. Route History

Add command to view recent analyses:

```python
async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /history command."""
    user_id = update.effective_user.id

    try:
        # Fetch user's routes from backend
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{API_URL}/api/v1/routes",
                params={"user_id": user_id, "limit": 10},
            )
            routes = response.json()["routes"]

        if not routes:
            await update.message.reply_text("You haven't analyzed any routes yet!")
            return

        # Format history
        history_text = "📋 Your Recent Analyses:\n\n"
        for i, route in enumerate(routes, 1):
            history_text += (
                f"{i}. Grade: {route['grade']} | "
                f"Date: {route['created_at'][:10]}\n"
                f"   ID: {route['id']}\n\n"
            )

        await update.message.reply_text(history_text)

    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        await update.message.reply_text("Failed to fetch history. Try again later.")

# In main():
application.add_handler(CommandHandler("history", history_command))
```

#### 4. Admin Commands

Add admin-only features:

```python
ADMIN_USER_IDS = [int(id) for id in os.getenv("TELEGRAM_ADMIN_USER_IDS", "").split(",") if id]

def is_admin(user_id: int) -> bool:
    """Check if user is admin."""
    return user_id in ADMIN_USER_IDS

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats command (admin only)."""
    if not is_admin(update.effective_user.id):
        await update.message.reply_text("❌ This command is for admins only.")
        return

    try:
        # Fetch stats from backend
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_URL}/api/v1/stats")
            stats = response.json()

        stats_text = (
            f"📊 Bot Statistics\n\n"
            f"Total Routes: {stats['total_routes']}\n"
            f"Total Users: {stats['total_users']}\n"
            f"Today: {stats['routes_today']}\n"
            f"This Week: {stats['routes_week']}\n"
        )

        await update.message.reply_text(stats_text)

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        await update.message.reply_text("Failed to fetch stats.")

# In main():
application.add_handler(CommandHandler("stats", stats_command))
```

---

## Deployment

### Option 1: Serverless (AWS Lambda)

**Pros**: Pay per use, auto-scaling, no server maintenance
**Cons**: Cold starts, limited execution time

#### Setup

1. **Install AWS CLI and SAM CLI**

2. **Create Lambda Function**:

```python
# lambda_handler.py
import json
import os
from telegram import Update
from telegram.ext import Application

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]

application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

# ... (add all your handlers here)

async def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        update = Update.de_json(json.loads(event["body"]), application.bot)
        await application.process_update(update)

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "OK"}),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
```

3. **Deploy**:

```bash
# Package dependencies
pip install -t package python-telegram-bot httpx
cd package && zip -r ../lambda.zip . && cd ..
zip -g lambda.zip lambda_handler.py

# Upload to AWS Lambda
aws lambda create-function \
    --function-name telegram-bot \
    --runtime python3.10 \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://lambda.zip \
    --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-role
```

4. **Set up API Gateway** to receive webhooks

5. **Set webhook**:

```python
import requests

TELEGRAM_BOT_TOKEN = "your-token"
WEBHOOK_URL = "https://your-api-gateway.amazonaws.com/webhook"

requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook",
    json={"url": WEBHOOK_URL}
)
```

### Option 2: Google Cloud Functions

Similar to AWS Lambda, but using Google Cloud.

### Option 3: Dedicated Server (VPS)

**Pros**: Full control, no cold starts, persistent connections
**Cons**: Server maintenance, fixed costs

#### Setup

1. **Get a VPS** (DigitalOcean, Linode, AWS EC2, etc.)

2. **Install Dependencies**:

```bash
sudo apt update
sudo apt install python3 python3-pip nginx certbot
```

3. **Clone Repository**:

```bash
git clone https://github.com/yourusername/telegram-bot.git
cd telegram-bot
pip3 install -r requirements.txt
```

4. **Set up Systemd Service**:

Create `/etc/systemd/system/telegram-bot.service`:

```ini
[Unit]
Description=Telegram Bouldering Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/telegram-bot
Environment="TELEGRAM_BOT_TOKEN=your-token"
Environment="TELEGRAM_API_URL=https://api.yourdomain.com"
ExecStart=/usr/bin/python3 bot.py
Restart=always

[Install]
WantedBy=multi-user.target
```

5. **Start Service**:

```bash
sudo systemctl daemon-reload
sudo systemctl start telegram-bot
sudo systemctl enable telegram-bot
sudo systemctl status telegram-bot
```

6. **Optional: Set up Webhook** (instead of polling):

Create a simple webhook server using FastAPI:

```python
# webhook_server.py
from fastapi import FastAPI, Request
from telegram import Update
from telegram.ext import Application

app = FastAPI()
application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

@app.post("/webhook")
async def webhook(request: Request):
    """Receive Telegram updates."""
    update = Update.de_json(await request.json(), application.bot)
    await application.process_update(update)
    return {"ok": True}
```

Run with:

```bash
uvicorn webhook_server:app --host 0.0.0.0 --port 8443
```

Set webhook:

```python
bot.set_webhook(url="https://yourdomain.com:8443/webhook")
```

### Option 4: Docker Container

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot.py .

CMD ["python", "bot.py"]
```

Build and run:

```bash
docker build -t telegram-bot .
docker run -d \
    -e TELEGRAM_BOT_TOKEN=your-token \
    -e TELEGRAM_API_URL=https://api.yourdomain.com \
    --name telegram-bot \
    telegram-bot
```

---

## User Guide

### For End Users

#### Getting Started

1. **Find the Bot**:
   - Open Telegram
   - Search for `@bouldering_analysis_bot` (or your bot's username)
   - Tap "Start" or send `/start`

2. **Send a Photo**:
   - Take a photo of a bouldering route
   - Send it to the bot (no caption needed)
   - Wait 5-10 seconds for analysis

3. **Get Results**:
   - Bot will reply with grade prediction
   - Includes confidence level and explanation
   - Option to provide feedback

#### Tips for Best Results

✅ **Do**:
- Include the entire route in the photo
- Use good lighting
- Take photo straight-on (not at an angle)
- Make sure holds are clearly visible

❌ **Don't**:
- Send blurry or dark photos
- Crop out parts of the route
- Include multiple routes in one photo

#### Example Conversation

```
You: /start

Bot: 🧗 Welcome to Bouldering Route Analyzer!
     Send me a photo of a climbing route...

You: [sends photo]

Bot: 🔍 Analyzing your route...
     This may take a few seconds.

     [5 seconds later]

Bot: ✅ Analysis Complete!

     📊 Grade Prediction: V5
     🎯 Confidence: 75%

     📝 Explanation:
     This route is graded V5 because:
     • Average hold quality: moderate (crimps and edges)
     • Max reach distance: 1.8m (above average)
     • Hold density: medium (11 holds total)
     • Wall angle: slightly overhanging

     Was this prediction accurate?
     [👍 Accurate] [👎 Not Accurate]

You: [taps 👍 Accurate]

Bot: ✅ Thanks for your feedback!
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Bot Not Responding

**Symptoms**: Bot doesn't reply to messages

**Solutions**:

1. **Check bot is running**:
   ```bash
   # If using systemd
   sudo systemctl status telegram-bot

   # If running manually
   ps aux | grep bot.py
   ```

2. **Check logs**:
   ```bash
   # Systemd logs
   sudo journalctl -u telegram-bot -f

   # Manual logs
   tail -f bot.log
   ```

3. **Verify token**:
   ```bash
   # Test token
   curl https://api.telegram.org/bot<YOUR_TOKEN>/getMe
   ```

4. **Restart bot**:
   ```bash
   sudo systemctl restart telegram-bot
   ```

#### Issue 2: Photo Upload Fails

**Symptoms**: Error when sending photos

**Solutions**:

1. **Check file size**:
   - Telegram limits: 20MB for bots
   - Your limit: Check `TELEGRAM_MAX_FILE_SIZE_MB`

2. **Check API connectivity**:
   ```python
   # Test API manually
   import httpx
   response = httpx.get(f"{API_URL}/health")
   print(response.status_code)  # Should be 200
   ```

3. **Check CORS settings** on backend

4. **Verify Supabase storage** is configured

#### Issue 3: Webhook Not Working

**Symptoms**: Updates not received in webhook mode

**Solutions**:

1. **Check webhook URL**:
   ```python
   import requests
   response = requests.get(
       f"https://api.telegram.org/bot{TOKEN}/getWebhookInfo"
   )
   print(response.json())
   ```

2. **Verify SSL certificate**:
   - Telegram requires HTTPS with valid SSL
   - Use Let's Encrypt for free SSL

3. **Check webhook is set**:
   ```python
   # Set webhook
   requests.post(
       f"https://api.telegram.org/bot{TOKEN}/setWebhook",
       json={"url": "https://yourdomain.com/webhook"}
   )
   ```

4. **Delete webhook** (fall back to polling):
   ```python
   requests.post(
       f"https://api.telegram.org/bot{TOKEN}/deleteWebhook"
   )
   ```

#### Issue 4: High Memory Usage

**Symptoms**: Bot uses excessive memory

**Solutions**:

1. **Restart bot regularly**:
   ```bash
   # Add to crontab
   0 3 * * * systemctl restart telegram-bot
   ```

2. **Limit concurrent requests**:
   ```python
   from telegram.ext import Application

   application = Application.builder() \
       .token(TOKEN) \
       .concurrent_updates(10) \  # Limit concurrent updates
       .build()
   ```

3. **Use webhook instead of polling**

4. **Monitor with `htop` or `prometheus`**

---

## Best Practices

### Security

- **Never commit bot token** to Git
- **Use environment variables** for all secrets
- **Validate user input** before processing
- **Rate limit** users to prevent abuse
- **Log suspicious activity**

### Performance

- **Use async/await** for all I/O operations
- **Implement caching** for frequent requests
- **Set reasonable timeouts** (30s max)
- **Handle errors gracefully**

### User Experience

- **Provide clear error messages**
- **Show progress indicators** for long operations
- **Use emoji** to make messages friendly
- **Keep messages concise** (Telegram works best with short messages)

---

## Additional Resources

- [python-telegram-bot Documentation](https://docs.python-telegram-bot.org/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Bot FAQ](https://core.telegram.org/bots/faq)
- [Example Bots](https://github.com/python-telegram-bot/python-telegram-bot/tree/master/examples)

---

**Questions or issues?** Open an issue in the GitHub repository or contact the maintainers.
