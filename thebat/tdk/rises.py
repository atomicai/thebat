import dotenv
from flask import Flask, redirect, url_for, render_template, request
from flask_session import Session
import logging
from pathlib import Path
import os

dotenv.load_dotenv()

# also we want to send cool inline buttons below, so we need to import:
from pytgbot.api_types.sendable.reply_markup import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup
from teleflask import Teleflask
from thebat.telegram.telestate import TeleState, machine
from thebat.telegram.telestate.contrib import SimpleDictDriver

# because we wanna send HTML formatted messages below, we need:
from teleflask.messages import HTMLMessage, TextMessage


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

app = Flask(
    __name__,
    template_folder="build",
    static_folder="build",
    root_path=Path(os.getcwd()) / "whooshai",
)

app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

app.config['CELERY_BROKER_URL'] = f'amqp://{os.environ["RABBIT_USERNAME"]}:{os.environ["RABBIT_PASSWORD"]}@localhost:5672'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

Session(app)


bot = Teleflask(api_key=os.environ.get("BOT_TOKEN"), app=app)

memo = SimpleDictDriver()

machine = machine.TeleStateMachine(__name__, database_driver=memo, teleflask_or_tblueprint=bot)


machine.ASKED_QUERY = TeleState("ASKED_QUERY", machine)
machine.CONFIRM_DATA = TeleState("CONFIRM_DATA", machine)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    return redirect(url_for('index'))


@machine.ALL.on_command("start")
def start(update, text):
    machine.set("ASKED_QUERY")
    return TextMessage(
        "<b>Hello! </b>This IR consisting of <u>\"search by embedding\"</u> and <b>BART</b> generation on top of it. Search anything related to Elon Musk.",
        parse_mode="html",
    )


@machine.ALL.command("cancel")
def cmd_cancel(update, text):
    old_action = machine.CURRENT
    machine.set("DEFAULT")
    if old_action == machine.DEFAULT:
        return TextMessage("Nothing to cancel.", parse_mode="text")
    # end if
    return TextMessage("All actions canceled.", parse_mode="text")


@machine.ASKED_QUERY.on_message("text")
def some_function(update, msg):
    query = msg.text.strip()
    # response_top_k = list(retriever.retrieve_top_k(query))[0][0]
    res
    assert len(response_top_k) == int(os.environ.get("TOPK", 5))
    # response = " ".join([x.text for x in response_top_k])
    response = "May the odds be ever in your favor"

    machine.set("CONFIRM_DATA", data={"query": query, "response": response})
    return HTMLMessage(
        f"<u>Query:</u> {escape(query)}\n---\n<u>Response:</u> {response}",
        reply_markup=InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("👌", callback_data="confirm_true"),
                ],
                [
                    InlineKeyboardButton('🤦', callback_data="confirm_false"),
                ],
            ]
        ),
    )
