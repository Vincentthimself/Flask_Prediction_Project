from flask import Flask, redirect, url_for
from Flask import create_app

app = create_app()

@app.route('/')
def index():
    return redirect(url_for('auth.register'))

if __name__ == '__main__':
    app.run(debug=True)