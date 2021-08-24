#ifndef WEB_SOCKET_CLIENT_H
#define WEB_SOCKET_CLIENT_H

#include <QObject>
#include <QAbstractSocket>

QT_FORWARD_DECLARE_CLASS(QWebSocket)
QT_FORWARD_DECLARE_CLASS(QTimer)


class WebSocketClient : public QObject
{
    Q_OBJECT
public:
    WebSocketClient(std::string& sURL, QObject *parent = nullptr);
    ~WebSocketClient();

    void CloseConnection();
    void SendFrame(int &nWidth, int nHeight, uint8_t *pData);
private:
    void Initialize( std::string& sURL );
    void DeInitialize();

private slots:
    void Connected();
    void Disconnected();
    void ProcessMessage(const QString &message);
    void OnError( QAbstractSocket::SocketError error);

signals:

private:
    QWebSocket*  m_pClentWebSocket;
    bool         m_bIsConnected;
    std::string  m_sURL;
    QTimer*      m_pConnectTimer;
    void*        m_pJpegCompressor;

};

#endif // WEB_SOCKET_CLIENT_H
