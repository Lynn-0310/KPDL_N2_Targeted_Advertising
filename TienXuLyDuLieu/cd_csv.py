import pandas as pd

# Đường dẫn tới các file .dat
movies_file = "D:/code/KPDL/BTL/data/movies.dat"
ratings_file = "D:/code/KPDL/BTL/data/ratings.dat"
users_file  = "D:/code/KPDL/BTL/data/users.dat"

# Đọc dữ liệu từ các file .dat với mã hóa phù hợp
ratings = pd.read_csv(ratings_file, sep="::", header=None, engine='python', 
                       names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin1")
users = pd.read_csv(users_file, sep="::", header=None, engine='python', 
                     names=["UserID", "Gender", "Age", "Occupation", "Zip-code"], encoding="latin1")
movies = pd.read_csv(movies_file, sep="::", header=None, engine='python', 
                      names=["MovieID", "Title", "Genres"], encoding="latin1")

# Lưu các file sang định dạng CSV
ratings.to_csv("D:/code/KPDL/BTL/TienXuLyDuLieu/ratings.csv", index=False)
users.to_csv("D:/code/KPDL/BTL/TienXuLyDuLieu/users.csv", index=False)
movies.to_csv("D:/code/KPDL/BTL/TienXuLyDuLieu/movies.csv", index=False)

print("Đã chuyển đổi các file .dat sang .csv:")
print("- ratings.csv")
print("- users.csv")
print("- movies.csv")
