import pandas as pd

# Đọc dữ liệu
df = pd.read_csv("D:/code/KPDL/BTL/TienXuLyDuLieu/merged_data.csv")


# In ra 20 dòng đầu tiên để kiểm tra dữ liệu
print(df.head(20))

df = df.drop(columns=["UserID","Zip-code", "Timestamp","Title", "MovieID"])

# Tạo danh sách tất cả các thể loại duy nhất
all_genres = set('|'.join(df['Genres']).split('|'))

# Tạo các cột tương ứng cho mỗi thể loại và đánh dấu 1 nếu phim có thể loại đó
for genre in all_genres:
    df[genre] = df['Genres'].apply(lambda x: 1 if genre in x else 0)

# Điền giá trị thiếu cho các cột số bằng trung bình
df["Age"] = df["Age"].fillna(df["Age"].mean())  
df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# Điền giá trị thiếu cho các cột phân loại bằng mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Occupation"] = df["Occupation"].fillna(df["Occupation"].mode()[0])

# Chuẩn hoá dữ liệu cột Age
def normalize_age(age):
    if age < 18:
        return 1
    elif 18 <= age <= 34:
        return 2
    elif 35 <= age <= 54:
        return 3
    else:
        return 4

df["Age"] = df["Age"].apply(normalize_age)

# Đổi gender thành  0 và 1 tương ứng với F và M
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})

df = df.drop(['Genres'], axis=1)

# In kết quả sau khi xử lý để kiểm tra
print(df.head(20))

# Lưu kết quả vào file mới
output_path = 'D:/code/KPDL/BTL/TienXuLyDuLieu/clean_data.csv'
df.to_csv(output_path, index=False)

# Thông báo đã lưu dữ liệu vào file
print(f"\nDữ liệu sau khi rời rạc hóa đã được lưu vào {output_path}")
