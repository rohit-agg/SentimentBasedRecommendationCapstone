from flask import Flask, Blueprint, render_template, request
from model import model

blueprint = Blueprint('main', __name__, template_folder='templates')

@blueprint.route('/', methods=['GET', 'POST'])
def home():
  error = None
  top_products = []

  if request.method == 'POST':
    username = request.form['username']
    error = model.check_username(username)

    if error is None:
      recommended_products = model.recommended_products(username, 20)
      if recommended_products is None:
        error = "No products can be recommended for {}".format(username)
      else:
        top_products = model.top_products(recommended_products, 5)
        print("top_products = {}".format(top_products))
  
  return render_template('index.html', 
            title = "Sentiment-based Recommendation System", form_data = request.form, 
            error = error, top_products = top_products)

def create_app():
  app = Flask(__name__)
  app.config.from_object('config.Config')

  app.register_blueprint(blueprint)

  return app

app = create_app()

if __name__ == "__main__":
  app.run(port=5001)