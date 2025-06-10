import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')

class HybridMovieRecommender:
    def __init__(self, data_path='ml-100k/', n_components=50):
        self.data_path = data_path
        self.n_components = n_components
        self.genre_en2ko = {
            "unknown": "기타", "Action": "액션", "Adventure": "어드벤처",
            "Animation": "애니메이션", "Children's": "아동/가족", "Comedy": "코미디",
            "Crime": "범죄", "Documentary": "다큐멘터리", "Drama": "드라마",
            "Fantasy": "판타지", "Film-Noir": "느와르", "Horror": "공포",
            "Musical": "뮤지컬", "Mystery": "미스터리", "Romance": "로맨스",
            "Sci-Fi": "SF(공상과학)", "Thriller": "스릴러", "War": "전쟁", "Western": "서부극"
        }
        self.genre_ko2en = {v: k for k, v in self.genre_en2ko.items()}
        
        # 초기화 시 데이터 로드 및 모델 학습
        self._load_data()
        self._prepare_data()
        self._train_model()
    
    def _load_data(self):
        """데이터 로딩"""
        try:
            self.ratings = pd.read_csv(
                f'{self.data_path}u.data', 
                sep='\t', 
                names=['user_id', 'movie_id', 'rating', 'timestamp']
            )
            self.movies = pd.read_csv(
                f'{self.data_path}u.item', 
                sep='|', 
                encoding='latin-1', 
                header=None,
                usecols=list(range(24)),
                names=[
                    'movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
                    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
                ]
            )
            print(f"데이터 로딩 완료: {len(self.ratings)}개 평점, {len(self.movies)}개 영화")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {e}")
    
    def _prepare_data(self):
        """데이터 전처리 및 인코딩"""
        # 인코더 초기화
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 사용자와 영화 ID 인코딩
        self.ratings['user_idx'] = self.user_encoder.fit_transform(self.ratings['user_id'])
        self.ratings['movie_idx'] = self.item_encoder.fit_transform(self.ratings['movie_id'])
        
        # 차원 정보
        self.n_users = self.ratings['user_idx'].nunique()
        self.n_items = self.ratings['movie_idx'].nunique()
        
        # Sparse matrix 생성 (메모리 효율성)
        self.rating_matrix = coo_matrix(
            (self.ratings['rating'], (self.ratings['user_idx'], self.ratings['movie_idx'])),
            shape=(self.n_users, self.n_items)
        ).tocsr()  # CSR format으로 변환 (연산 효율성)
        
        # 사용자별 시청 이력 캐싱
        self._cache_user_history()
        
        print(f"매트릭스 크기: {self.n_users} x {self.n_items}")
        print(f"Sparsity: {(1 - self.rating_matrix.nnz / (self.n_users * self.n_items)) * 100:.2f}%")
    
    def _cache_user_history(self):
        """사용자별 시청 이력 캐싱 (성능 최적화)"""
        self.user_watched = {}
        for user_id in self.ratings['user_id'].unique():
            watched_movies = self.ratings[self.ratings['user_id'] == user_id]['movie_id'].values
            self.user_watched[user_id] = set(watched_movies)
    
    def _train_model(self):
        """SVD 모델 학습 및 예측 매트릭스 생성"""
        print("SVD 모델 학습 중...")
        self.svd = TruncatedSVD(n_components=self.n_components, n_iter=20, random_state=42)
        self.user_factors = self.svd.fit_transform(self.rating_matrix)
        
        # 전체 예측 매트릭스를 한 번에 계산 (벡터화 연산)
        self.predicted_ratings = self.user_factors @ self.svd.components_
        print("모델 학습 완료!")
    
    def _validate_user(self, user_id):
        """사용자 ID 유효성 검사"""
        if user_id not in self.ratings['user_id'].values:
            raise ValueError(f"사용자 ID {user_id}가 존재하지 않습니다.")
    
    def _normalize_genre(self, genre_input):
        """장르 입력 정규화"""
        genre_input = genre_input.strip()
        if genre_input in self.genre_ko2en:
            return self.genre_ko2en[genre_input]
        elif genre_input in self.genre_en2ko:
            return genre_input
        else:
            available_genres = list(self.genre_en2ko.keys()) + list(self.genre_ko2en.keys())
            raise ValueError(f"지원하지 않는 장르입니다. 사용 가능한 장르: {', '.join(available_genres)}")
    
    def get_user_predictions(self, user_id):
        """특정 사용자의 모든 영화에 대한 예측 점수 반환"""
        self._validate_user(user_id)
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            return self.predicted_ratings[user_idx]
        except ValueError:
            raise ValueError(f"사용자 ID {user_id}에 대한 예측을 생성할 수 없습니다.")
    
    def recommend_movies_all(self, user_id, top_n=10):
        """전체 영화 추천"""
        self._validate_user(user_id)
        
        # 사용자의 모든 예측 점수 가져오기
        user_predictions = self.get_user_predictions(user_id)
        
        # 이미 본 영화 제외
        watched_movies = self.user_watched.get(user_id, set())
        
        # 영화별 예측 점수와 함께 추천 리스트 생성
        recommendations = []
        for movie_id in self.movies['movie_id'].unique():
            if movie_id not in watched_movies:
                try:
                    movie_idx = self.item_encoder.transform([movie_id])[0]
                    score = user_predictions[movie_idx]
                    recommendations.append((movie_id, score))
                except ValueError:
                    continue  # 인코딩되지 않은 영화 건너뛰기
        
        # 점수 기준 정렬 후 상위 N개 반환
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    def recommend_movies_by_genre(self, user_id, genre_input, top_n=10):
        """장르별 영화 추천"""
        self._validate_user(user_id)
        genre = self._normalize_genre(genre_input)
        
        # 해당 장르의 영화만 필터링
        genre_movies = self.movies[self.movies[genre] == 1]['movie_id'].unique()
        
        # 사용자의 모든 예측 점수 가져오기
        user_predictions = self.get_user_predictions(user_id)
        
        # 이미 본 영화 제외
        watched_movies = self.user_watched.get(user_id, set())
        
        # 장르 내 추천 리스트 생성
        recommendations = []
        for movie_id in genre_movies:
            if movie_id not in watched_movies:
                try:
                    movie_idx = self.item_encoder.transform([movie_id])[0]
                    score = user_predictions[movie_idx]
                    recommendations.append((movie_id, score))
                except ValueError:
                    continue
        
        return sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_movie_info(self, movie_id):
        """영화 정보 조회"""
        movie_info = self.movies[self.movies['movie_id'] == movie_id]
        if movie_info.empty:
            return None
        return movie_info.iloc[0]
    
    def print_recommendations(self, recommendations, title="추천 영화"):
        """추천 결과 출력"""
        print(f"\n=== {title} ===")
        if not recommendations:
            print("추천할 영화가 없습니다.")
            return
        
        for i, (movie_id, score) in enumerate(recommendations, 1):
            movie_info = self.get_movie_info(movie_id)
            if movie_info is not None:
                print(f"{i:2d}. {movie_info['title']} (예측 점수: {score:.2f})")
    
    def get_user_stats(self, user_id):
        """사용자 통계 정보"""
        self._validate_user(user_id)
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        stats = {
            'total_ratings': len(user_ratings),
            'avg_rating': user_ratings['rating'].mean(),
            'rating_std': user_ratings['rating'].std(),
            'favorite_genres': self._get_user_favorite_genres(user_id)
        }
        return stats
    
    def _get_user_favorite_genres(self, user_id):
        """사용자가 선호하는 장르 분석"""
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        high_rated_movies = user_ratings[user_ratings['rating'] >= 4]['movie_id']
        
        genre_counts = {}
        genre_columns = list(self.genre_en2ko.keys())
        
        for movie_id in high_rated_movies:
            movie_genres = self.movies[self.movies['movie_id'] == movie_id]
            if not movie_genres.empty:
                for genre in genre_columns:
                    if movie_genres.iloc[0][genre] == 1:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # 상위 3개 장르 반환
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        return [(self.genre_en2ko[genre], count) for genre, count in sorted_genres]

def main():
    """메인 실행 함수"""
    try:
        # 추천 시스템 초기화
        recommender = HybridMovieRecommender()
        
        # 사용자 입력
        user_id = int(input("사용자 ID를 입력하세요 (예: 1): "))
        
        # 사용자 통계 출력
        stats = recommender.get_user_stats(user_id)
        print(f"\n=== 사용자 {user_id} 통계 ===")
        print(f"총 평점 수: {stats['total_ratings']}")
        print(f"평균 평점: {stats['avg_rating']:.2f}")
        print(f"선호 장르: {', '.join([f'{genre}({count})' for genre, count in stats['favorite_genres']])}")
        
        # 추천 방식 선택
        print("\n추천 방식 선택:")
        print("[1] 전체 추천")
        print("[2] 장르별 추천")
        print("[3] 둘 다 보기")
        
        mode = input("번호를 입력하세요 (1, 2, 또는 3): ").strip()
        
        if mode in ["1", "3"]:
            recommendations = recommender.recommend_movies_all(user_id, top_n=10)
            recommender.print_recommendations(recommendations, f"사용자 {user_id}의 전체 추천 영화")
        
        if mode in ["2", "3"]:
            print(f"\n지원 장르(영어): {', '.join(recommender.genre_en2ko.keys())}")
            print(f"지원 장르(한글): {', '.join(recommender.genre_ko2en.keys())}")
            genre_input = input("추천받고 싶은 장르를 입력하세요 (예: Comedy 또는 코미디): ")
            
            recommendations = recommender.recommend_movies_by_genre(user_id, genre_input, top_n=10)
            recommender.print_recommendations(recommendations, f"사용자 {user_id}의 '{genre_input}' 장르 추천 영화")
        
        if mode not in ["1", "2", "3"]:
            print("잘못된 입력입니다.")
    
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main()