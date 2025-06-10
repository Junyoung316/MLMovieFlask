from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from hybrid_movie_recommender import HybridMovieRecommender
import traceback

app = Flask(__name__)
CORS(app)  # React 앱에서의 요청을 허용

# 추천 시스템 초기화 (전역으로 한 번만 초기화)
try:
    recommender = HybridMovieRecommender()
    print("추천 시스템 초기화 완료!")
except Exception as e:
    print(f"추천 시스템 초기화 실패: {e}")
    recommender = None

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인"""
    return jsonify({
        'status': 'healthy',
        'recommender_ready': recommender is not None
    })

@app.route('/api/users/<int:user_id>/stats', methods=['GET'])
def get_user_stats(user_id):
    """사용자 통계 정보 조회"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        stats = recommender.get_user_stats(user_id)
        return jsonify({
            'user_id': user_id,
            'stats': stats
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/users/<int:user_id>/recommendations', methods=['GET'])
def get_recommendations(user_id):
    """전체 영화 추천"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        top_n = request.args.get('top_n', 10, type=int)
        recommendations = recommender.recommend_movies_all(user_id, top_n=top_n)
        
        # 영화 정보와 함께 반환
        detailed_recommendations = []
        for movie_id, score in recommendations:
            movie_info = recommender.get_movie_info(movie_id)
            if movie_info is not None:
                detailed_recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'score': float(round(score, 2)),
                    'release_date': movie_info['release_date'] if movie_info['release_date'] else 'Unknown',
                    'genres': get_movie_genres(movie_info)
                })
        
        return jsonify({
            'user_id': user_id,
            'recommendations': detailed_recommendations,
            'count': len(detailed_recommendations)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"추천 오류: {traceback.format_exc()}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/users/<int:user_id>/recommendations/genre/<genre>', methods=['GET'])
def get_genre_recommendations(user_id, genre):
    """장르별 영화 추천"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        top_n = request.args.get('top_n', 10, type=int)
        recommendations = recommender.recommend_movies_by_genre(user_id, genre, top_n=top_n)
        
        # 영화 정보와 함께 반환
        detailed_recommendations = []
        for movie_id, score in recommendations:
            movie_info = recommender.get_movie_info(movie_id)
            if movie_info is not None:
                detailed_recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'score': float(round(score, 2)),
                    'release_date': movie_info['release_date'] if movie_info['release_date'] else 'Unknown',
                    'genres': get_movie_genres(movie_info)
                })
        
        return jsonify({
            'user_id': user_id,
            'genre': genre,
            'recommendations': detailed_recommendations,
            'count': len(detailed_recommendations)
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"장르 추천 오류: {traceback.format_exc()}")
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/genres', methods=['GET'])
def get_genres():
    """지원하는 장르 목록 반환"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        return jsonify({
            'genres_english': list(recommender.genre_en2ko.keys()),
            'genres_korean': list(recommender.genre_ko2en.keys()),
            'genre_mapping': recommender.genre_en2ko
        })
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/movies/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    """영화 상세 정보 조회"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        movie_info = recommender.get_movie_info(movie_id)
        if movie_info is None:
            return jsonify({'error': '영화를 찾을 수 없습니다.'}), 404
        
        return jsonify({
            'movie_id': movie_id,
            'title': movie_info['title'],
            'release_date': movie_info['release_date'] if movie_info['release_date'] else 'Unknown',
            'imdb_url': movie_info['IMDb_URL'] if movie_info['IMDb_URL'] else '',
            'genres': get_movie_genres(movie_info)
        })
        
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

@app.route('/api/users', methods=['GET'])
def get_user_list():
    """사용 가능한 사용자 ID 목록 반환 (처음 20명)"""
    try:
        if not recommender:
            return jsonify({'error': '추천 시스템이 초기화되지 않았습니다.'}), 500
        
        user_ids = sorted(recommender.ratings['user_id'].unique())[:20]
        return jsonify({
            'user_ids': user_ids.tolist(),
            'total_users': len(recommender.ratings['user_id'].unique())
        })
        
    except Exception as e:
        return jsonify({'error': f'서버 오류: {str(e)}'}), 500

def get_movie_genres(movie_info):
    """영화의 장르 목록 추출"""
    if not recommender:
        return []
    
    genres = []
    for genre_en, genre_ko in recommender.genre_en2ko.items():
        if movie_info[genre_en] == 1:
            genres.append({
                'english': genre_en,
                'korean': genre_ko
            })
    return genres

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API 엔드포인트를 찾을 수 없습니다.'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '내부 서버 오류가 발생했습니다.'}), 500

if __name__ == '__main__':
    print("Flask 서버 시작...")
    app.run(debug=True, host='0.0.0.0', port=5521)
