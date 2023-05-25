import discord
import os
import threading
import asyncio
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()

print('Loading AI model...')
tokenizer = AutoTokenizer.from_pretrained('tinkoff-ai/ruDialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('tinkoff-ai/ruDialoGPT-medium')

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

def getQuery(message: discord.Message) -> str:
    query = f'@@ПЕРВЫЙ@@{message.content}@@ВТОРОЙ@@'

    ref = message.reference
    if not ref:
        return query

    if not ref.cached_message:
        return query
    
    if not ref.cached_message.author.id == client.user.id:
        return query

    ref2 = ref.cached_message.reference
    if not ref2:
        return query

    if not ref2.cached_message:
        return query
    
    if not ref2.cached_message.author.id == message.author.id:
        return query

    return f'@@ПЕРВЫЙ@@{ref2.cached_message.content}@@ВТОРОЙ@@{ref.cached_message.content}' + query

async def generateReply(message: discord.Message):
    query = getQuery(message)

    async with message.channel.typing():
        inputs = tokenizer(query, return_tensors='pt')
        generated_token_ids = model.generate(
            **inputs,
            top_k=10,
            top_p=0.95,
            num_beams=3,
            num_return_sequences=3,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=1.2,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40
        )
        context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
        response = context_with_response[0].replace(query, '').split('@@ПЕРВЫЙ@@')[0]

        await message.reply(response)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    if message.channel.id == int(os.getenv('CHANNEL_ID')):
        await generateReply(message)

client.run(os.getenv('BOT_TOKEN'))