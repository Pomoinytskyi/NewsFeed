import asyncio
import logging
import nltk
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from goose3 import Goose
import torch

from nltk.tokenize import sent_tokenize, word_tokenize
from time import perf_counter


summarizer = ""

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logging.getLogger('apscheduler').setLevel(logging.WARNING)
logging.getLogger('telegram._bot').setLevel(logging.WARNING)
logging.getLogger('httpx._client').setLevel(logging.WARNING)
logging.getLogger('telegram.ext._application').setLevel(logging.WARNING)
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="I'm a Summary bot, Send me text for summarization.")

async def createSummary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    inputText=update.message.text    
    logging.debug( f"Received Mesage from {update.effective_chat.id}\n\t {inputText}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Processing...")
    summary = await getSummary(inputText)
    
    logging.debug( f"Summary\n\t{summary}")
    await context.bot.send_message(chat_id=update.effective_chat.id, text=summary, device = 1)

async def getSummary(inputText):
    inputTextLength = len(inputText)
    summaryMaxLength = min(int(inputTextLength/2), 300)
    summaryMinLenght= min(int(summaryMaxLength/2), 200)

    t1_start = perf_counter()

    summaryObject = summarizer(inputText, max_length=summaryMaxLength, min_length=summaryMinLenght, do_sample=False)
    t1_stop = perf_counter()
    logging.info(f"Duration: {t1_stop - t1_start}")

    summary = summaryObject[0]['summary_text']
    return summary

async def parseUrl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    asyncio.create_task(
        processParseUrl(update, context))

async def processParseUrl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    url = update.message.text.removeprefix('/url ')
    logging.debug( f"Received URL from {update.effective_chat.id}\n\t {url}")
    g = Goose()
    article = g.extract(url=url)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Processing '{article.title}'")

    article = article.cleaned_text
    sentences = sent_tokenize(article)
    
    batches = []
    currentBatchSize = 0;
    batch = ''
    for sentence in sentences:
        if(len(sentence) >= 1024):
            continue
        if(currentBatchSize + len(sentence) + 1 < 1024):
            batch = batch + ' ' + sentence
            currentBatchSize += len(sentence) + 1
        else:
            batches.append(batch)
            batch = sentence
            currentBatchSize = len(batch)

    summaries = []
    for batch in batches:
        summary = await getSummary(batch)
        summaries.append(summary)
        
    for summary in summaries:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=summary)

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Done.")

if __name__ == '__main__':
    logging.debug( f"Models Initialization...")
    #tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    #model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=torch.cuda.current_device())
    
    logging.debug( f"Initialized.")
    
    application = ApplicationBuilder().token('< Telegram bot tocken >').build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    # parseUrl_handler = CommandHandler('url', parseUrl)
    # application.add_handler(parseUrl_handler)

    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), parseUrl)
    application.add_handler(echo_handler)

    application.run_polling()