import requests
from flask import Flask, request

from .db import cursor

app = Flask(__name__)


@app.route("/employee")
def employee():
    name = request.args.get("name", "")
    cursor.execute(f"SELECT * FROM employees WHERE name = '{name}'")
    return {"rows": cursor.fetchall()}


@app.route("/sync")
def sync():
    return requests.get("https://payroll.internal/export", verify=False).json()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
