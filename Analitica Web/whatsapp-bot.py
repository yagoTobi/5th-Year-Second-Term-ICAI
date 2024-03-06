import discord

public_key = "527cb717ec9256d95ac7e5a329bfd19275a53e26d02ccc5d488a7d9a81630b6b"
application_id = "1201563077372563618"

client = discord.Client()

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

client.run('your token here')
