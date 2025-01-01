import pandas as pd

# Đường dẫn tới các file CSV
ratings_file = "D:/code/KPDL/BTL/TienXuLyDuLieu/ratings.csv"
users_file = "D:/code/KPDL/BTL/TienXuLyDuLieu/users.csv"
movies_file = "D:/code/KPDL/BTL/TienXuLyDuLieu/movies.csv"

# Đọc dữ liệu từ các file CSV
ratings = pd.read_csv(ratings_file)
users = pd.read_csv(users_file)
movies = pd.read_csv(movies_file)

# Kết hợp các bảng dữ liệu
merged = pd.merge(ratings, users, on="UserID")
merged = pd.merge(merged, movies, on="MovieID")

# Loại bỏ các cột trùng lặp và sắp xếp lại cột
merged = merged[["UserID", "Gender", "Age", "Occupation", "Zip-code", "MovieID", "Title", "Genres", "Rating", "Timestamp"]]

# Lưu kết quả vào file CSV mới
output_file = "D:/code/KPDL/BTL/TienXuLyDuLieu/merged_data.csv"
merged.to_csv(output_file, index=False)

print(f"Dữ liệu đã được ghép nối và lưu vào: {output_file}")
