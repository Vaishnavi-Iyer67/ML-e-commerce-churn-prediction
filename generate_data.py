#basic code I have used 
import pandas as pd

# Create a simple churn dataset
data = {
    "Age": [25,40,31,22,55,30,28,45,33,27,50,26,34,29,48,32,23,37,41,36],
    "TotalOrders": [10,2,6,1,3,8,4,9,2,5,1,7,3,6,2,9,1,4,8,5],
    "LastPurchaseDaysAgo": [5,60,18,90,45,12,35,7,70,20,85,10,40,22,65,14,95,30,9,25],
    "Churn": [0,1,0,1,1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0]
}

df = pd.DataFrame(data)

# Save dataset as CSV
df.to_csv("churn_data.csv", index=False)

print("Dataset created successfully!")
print(df)