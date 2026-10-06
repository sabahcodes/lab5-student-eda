import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Create the dataset (12 rows: 2 missing Marks, 1 duplicate) ----------
data = {
    "StudentID":     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 5],
    "Gender":        ["Male", "Female", "Male", "Female", "Male", "Female",
                      "Male", "Female", "Male", "Female", "Female", "Male"],
    "StudyHours":    [5, 8, 3, 9, 6, 7, 2, 10, 4, 8, 6, 6],
    "Attendance(%)": [80, 92, 70, 95, 85, 90, 65, 98, 75, 88, 82, 85],
    "Marks":         [65, 82, np.nan, 88, 68, 79, 45, 91, np.nan, 84, 70, 68],
}
pd.DataFrame(data).to_csv("student_performance.csv", index=False)

# ---------- (a) Load the CSV, print shape and info ----------
df = pd.read_csv("student_performance.csv")
print("Shape:", df.shape)
print(df.info())

# ---------- (b) Missing values, duplicates, cleaning ----------
print("Missing values per column:")
print(df.isnull().sum())
print("Missing values in Marks:", df["Marks"].isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()
df["Marks"] = df["Marks"].fillna(df["Marks"].median())
print("After cleaning:", df.shape)

# ---------- (c) Histogram and boxplot of Marks ----------
plt.figure(figsize=(6, 4))
plt.hist(df["Marks"], bins=6, color="steelblue", edgecolor="black")
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Marks Distribution")
plt.tight_layout()
plt.savefig("marks_histogram.png")
plt.close()

plt.figure(figsize=(4, 5))
plt.boxplot(df["Marks"])
plt.ylabel("Marks")
plt.title("Marks Boxplot")
plt.tight_layout()
plt.savefig("marks_boxplot.png")
plt.close()

# ---------- (d) Scatter plot and correlation matrix ----------
plt.figure(figsize=(6, 4))
plt.scatter(df["StudyHours"], df["Marks"], color="darkorange")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs. Marks")
plt.tight_layout()
plt.savefig("studyhours_vs_marks.png")
plt.close()

print("Correlation matrix:")
print(df[["StudyHours", "Attendance(%)", "Marks"]].corr())

# ---------- (e) Compare genders with groupby ----------
print("Average Marks and Attendance by Gender:")
print(df.groupby("Gender")[["Marks", "Attendance(%)"]].mean())

print("Saved marks_histogram.png, marks_boxplot.png, studyhours_vs_marks.png")