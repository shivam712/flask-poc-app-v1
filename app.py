from flask import Flask, redirect, url_for,request, jsonify
import pandas as pds
from flask_jsonpify import jsonpify
import Protection1
import Protection2
import DataEntry2 as de

app = Flask(__name__)

@app.route("/")
def home():
    return "This is the main page! <h1>I am redirecting from home function<h1>"

@app.route("/<name>")
def user(name):
    return f"Hi I am taking input from user {name}!"

@app.route("/admin")
def admin():
    redirect(url_for("home"))

@app.route("/modelcalc", methods=['GET'])
def calc():
    if 'protection' in request.args:
        protection = int(request.args['protection'])
    else:
        print(fun.protection1())
        return "Error: No id field provided. Please specify Protection=1 or Protection=2."
        
    if protection == 1:
        # obj_protection_1 = Protection1.tablesProtection1()
        # df_list = obj_protection_1.finalTableProtection1().values.tolist()
        df_list = de.protection1().values.tolist()
        return jsonify(df_list)
    if protection == 2:
        # obj_protection_2 = Protection2.tablesProtection2()
        # df_list = obj_protection_2.finalTableProtection2().values.tolist()
        df_list = de.protection2().values.tolist()
        return jsonify(df_list)



if __name__ =="__main__":
    app.run
    
