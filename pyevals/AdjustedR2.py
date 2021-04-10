import pandas as pd
def AdjustedR2(R2_score,x_train):
	N = len(x_train)
	IndependentFeatures = pd.DataFrame(x_train)
	p = len(IndependentFeatures.columns)

	Numerator = R2_score*(N-1)
	Denominator = (N-p-1)

	Ar2 = 1 - (Numerator/Denominator)

	return Ar2
