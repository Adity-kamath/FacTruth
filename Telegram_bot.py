import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

print("DEBUG - cwd:", os.getcwd())
print("DEBUG - token found:", repr(os.getenv("TELEGRAM_BOT_TOKEN")))

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from core import run_fact_check


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(
        "👋 Welcome to facTruth!\n\n"
        "Send me a claim, headline, or forwarded message "
        "and I will check it against live sources.\n\n"
        "Example:\n"
        "Is the Earth flat?"
    )


async def verify_claim(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    incoming_text = (
        update.message.text or ""
    ).strip()

    if not incoming_text:

        await update.message.reply_text(
            "Please send a claim or message to verify."
        )

        return

    # Tell the user that verification has started
    processing_message = await update.message.reply_text(
        "🔎 Checking live sources..."
    )

    try:

        result = run_fact_check(
            incoming_text
        )

        if "error" in result:

            await processing_message.edit_text(
                "⚠️ " + str(result["error"])
            )

            return

        verdict = result.get(
            "verdict",
            "Unverified"
        )

        confidence = result.get(
            "confidence",
            0
        )

        summary = result.get(
            "summary",
            ""
        )

        lines = [

            f"🔎 *facTruth verdict*",

            "",

            f"*Verdict:* {verdict}",

            f"*Confidence:* {confidence}%",

            "",

            f"*Explanation:*\n{summary}",
        ]


        contradictions = result.get(
            "contradictions",
            []
        )

        if contradictions:

            lines.extend(
                [
                    "",
                    "*⚠️ Contradictions:*",
                ]
            )

            for item in contradictions[:3]:

                lines.append(
                    f"• {item}"
                )


        sources = result.get(
            "sources_used",
            []
        )

        if sources:

            lines.extend(
                [
                    "",
                    "*📚 Sources:*",
                ]
            )

            for source in sources[:3]:

                title = source.get(
                    "title",
                    "Source"
                )

                link = source.get(
                    "link",
                    ""
                )

                if link:

                    lines.append(
                        f"• [{title}]({link})"
                    )

                else:

                    lines.append(
                        f"• {title}"
                    )


        message = "\n".join(lines)

        # Telegram messages have a size limit.
        # Keep a safe limit for our bot.
        message = message[:3800]

        await processing_message.edit_text(
            message,
            parse_mode="Markdown",
            disable_web_page_preview=True,
        )

    except Exception as e:

        print(
            "Telegram verification error:",
            e
        )

        await processing_message.edit_text(
            "⚠️ Something went wrong while "
            "checking the claim."
        )


def main():

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable "
            "is not set."
        )

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            verify_claim
        )
    )

    print(
        "facTruth Telegram bot is running..."
    )

    application.run_polling(stop_signals=None)


if __name__ == "__main__":

    main()
