import socket, json
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 9000))
print("Escuchando alertas UDP en el puerto 9000 ...")
while True:
    datos, origen = s.recvfrom(65535)
    try:
        print(f"\n[{origen[0]}] {json.dumps(json.loads(datos), indent=2, ensure_ascii=False)}")
    except Exception:
        print(f"\n[{origen[0]}] {datos!r}")
