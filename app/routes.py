from flask import Blueprint, render_template, request
from app.model import model

blueprint = Blueprint('main', __name__, template_folder='templates')

@blueprint.route('/', methods=['GET', 'POST'])
def home():
  error = None
  top_products = []
  
  if request.method == 'POST':
    username = request.form['username']
    error = model.check_username(username)

    if error == None:
      recommended_products = model.recommended_products(username, 20)
      top_products = model.top_products(recommended_products, 5)
      print(top_products)
  
  return render_template('index.html', 
            title = "Sentiment-based Recommendation System", form_data = request.form, 
            error = error, top_products = top_products)