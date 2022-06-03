from joblib import dump,load
model=load('FinalPredictor.joblib')

print("Enter Following Parameters In Sequence")
print("CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	TAXRM")

features=[]
for i in range(0,13):
    x=float(input())
    features.append(x)
print(features)
print("Predicted MEDV value for provided Data Is :")
print(model.predict([features]))




