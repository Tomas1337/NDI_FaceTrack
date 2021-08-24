#include "WebSocketClient.h"
#include <QWebSocket>
#include "QImage"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "QTimer"
#include "turbojpeg.h"
#include "Base64Encode.h"

WebSocketClient::WebSocketClient(std::string& sURL, QObject *parent)
    : QObject(parent)
    , m_pClentWebSocket( nullptr )
    , m_bIsConnected( false )
    , m_pConnectTimer( nullptr )
    , m_pJpegCompressor( nullptr )
{
    Initialize( sURL );
    m_pJpegCompressor = tjInitCompress();
}

WebSocketClient::~WebSocketClient()
{
    if( nullptr != m_pJpegCompressor )
        tjDestroy(m_pJpegCompressor);
    DeInitialize();
}

void WebSocketClient::Initialize( std::string& sURL )
{
    DeInitialize();
    m_sURL = sURL;
    m_pClentWebSocket = new QWebSocket();
    if( nullptr != m_pClentWebSocket )
    {
        connect(m_pClentWebSocket, &QWebSocket::connected,
             this, &WebSocketClient::Connected);
        connect(m_pClentWebSocket, &QWebSocket::textMessageReceived,
             this, &WebSocketClient::ProcessMessage);
        connect(m_pClentWebSocket, &QWebSocket::disconnected,
             this, &WebSocketClient::Disconnected);
        connect(m_pClentWebSocket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error),
                this, &WebSocketClient::OnError);
        m_pClentWebSocket->open( QUrl( sURL.c_str() ));
    }
}

void WebSocketClient::DeInitialize()
{
    if( nullptr != m_pClentWebSocket )
    {
        if( m_bIsConnected )
            CloseConnection();
       //disconnect(m_pClentWebSocket, &QWebSocket::connected,
       //        this, &WebSocketClient::Connected);
       //disconnect(m_pClentWebSocket, &QWebSocket::textMessageReceived,
       //        this, &WebSocketClient::ProcessMessage);
       //disconnect(m_pClentWebSocket, &QWebSocket::disconnected,
       //        this, &WebSocketClient::Disconnected);
       //disconnect(m_pClentWebSocket, QOverload<QAbstractSocket::SocketError>::of(&QWebSocket::error),
       //        this, &WebSocketClient::OnError);
        delete m_pClentWebSocket;
        m_pClentWebSocket = nullptr;
    }
}

void WebSocketClient::Connected()
{
    m_bIsConnected = true;
}

void WebSocketClient::Disconnected()
{
    m_bIsConnected = false;
}

void WebSocketClient::ProcessMessage(const QString &message)
{
    qDebug() << message;
}

void WebSocketClient::OnError(QAbstractSocket::SocketError error)
{
        qDebug() << "Track Client Error Code : " << error << m_pClentWebSocket->errorString();
}

void WebSocketClient::CloseConnection()
{
    if( m_bIsConnected && ( nullptr != m_pClentWebSocket))
    {
        m_pClentWebSocket->close();
        m_bIsConnected = false;
    }
}

constexpr int STD_FRAME_WIDTH   = 640;
constexpr int STD_FRAME_HEIGHT  = 360;
constexpr int JPEG_QUALITY_80   = 80;
constexpr char JPEG_ENCODE[]    = ".jpeg";
constexpr char IMAGE_FRAME_HEADER[] = "image";

void WebSocketClient::SendFrame( int& nWidth, int nHeight, uint8_t* pData)
{
    if( m_bIsConnected )
    {
        QByteArray buffer;
        QDataStream trackStream(&buffer, QIODevice::WriteOnly);
        cv::Mat imgIn( nHeight, nWidth, CV_8UC4, pData);
        resize(imgIn,imgIn,cv::Size( STD_FRAME_WIDTH, STD_FRAME_HEIGHT));
        std::vector<uchar> imageBuffer;

        unsigned char* pCompressedImage = nullptr;
        long unsigned int nEnCodedJpegSize = 0;
        tjCompress2(m_pJpegCompressor, imgIn.data, nWidth, 0, nHeight, TJPF_RGBX,
                  &pCompressedImage, &nEnCodedJpegSize, TJSAMP_444, JPEG_QUALITY_80,
                  TJFLAG_FASTDCT);
        trackStream << QString(IMAGE_FRAME_HEADER);
        qDebug() << "Commressed length " << nEnCodedJpegSize;

        trackStream << writeRawData(( char*)pCompressedImage, nEnCodedJpegSize );

        int nLenth = 0;
        if( m_pClentWebSocket )
        {
            nLenth = m_pClentWebSocket->sendBinaryMessage( buffer );
            qDebug() << "Track Frame length " << nLenth;
        }
    }

}

//void WebSocketClient::StartConnectTimer()
//{
//    if( nullptr == m_pConnectTimer )
//    {
//        constexpr int BLOCK_UPDATE_TIME_GAP = 750;
//        m_pConnectTimer =  new QTimer(this);
//        QObject::connect(m_pConnectTimer, &QTimer::timeout, this, &WebSocketClient::);
//        m_pConnectTimer->start(BLOCK_UPDATE_TIME_GAP);
//    }
//}
//
//void WebSocketClient::StopConnectTimer()
//{
//    if( nullptr != m_pConnectTimer )
//    {
//        m_pConnectTimer->stop();
//        delete m_pConnectTimer;
//        m_pConnectTimer = nullptr;
//    }
//}
