from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Ensure the static folder is served for template images
app._static_folder = os.path.abspath("static/")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview', methods=['POST'])
def preview():
    # Get user input from the form
    name = request.form.get('name')
    event = request.form.get('event')
    date = request.form.get('date')

    # Render the preview with the provided data
    return render_template('preview.html', name=name, event=event, date=date)

if __name__ == "__main__":
    app.run(debug=True)
