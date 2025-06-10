내부 배포:
    
    python3 -m http.server 8080 (htt://localhost:8080)

외부 접속 배포:
    
    python3 -m http.server 8080 --bind 0.0.0.0 (http://호스트IP:8080)

    같은 네트워크 상에 있어야 접속 가능
    호스트 IP 확인(MAC OS):

        ifconfig | grep inet