#include <armadillo>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include "compute/edflib.h"
#include "compute/eeg_spectrogram.hpp"
#include "compute/eeg_change_point.hpp"
#include "json11/json11.hpp"
#include "wslib/server_ws.hpp"

using namespace arma;
using namespace std;
using namespace SimpleWeb;
using namespace json11;
using namespace boost::property_tree;

#define NUM_THREADS 4
#define PORT 8080
#define TEXT_OPCODE 129
#define BINARY_OPCODE 130

const char* CH_NAME_MAP[] = {"LL", "LP", "RP", "RL"};

void send_message(SocketServer<WS>* server, shared_ptr<SocketServer<WS>::Connection> connection,
                  string msg_type, ptree content, float* data, size_t data_size)
{

  ptree msg;
  msg.put("type", msg_type);
  msg.add_child("content", content);
  ostringstream header_stream;
  write_json(header_stream, msg);
  string header = header_stream.str();

  uint32_t header_len = header.size() + (8 - ((header.size() + 4) % 8));
  // append enough spaces so that the payload starts at an 8-byte
  // aligned position. The first four bytes will be the length of
  // the header, encoded as a 32 bit signed integer:
  header.resize(header_len, ' ');

  stringstream data_ss;
  data_ss.write((char*) &header_len, sizeof(uint32_t));
  data_ss.write(header.c_str(), header_len);
  if (data != NULL)
  {
    data_ss.write((char*) data, data_size);
  }

  // server.send is an asynchronous function
  server->send(connection, data_ss, [](const boost::system::error_code & ec)
  {
    if (ec)
    {
      cout << "Server: Error sending message. " <<
           // See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
           // Error Codes for error code meanings
           "Error: " << ec << ", error message: " << ec.message() << endl;
    }
  }, BINARY_OPCODE);
}

void log_json(ptree content)
{
  ostringstream content_stream;
  write_json(content_stream, content);
  cout << "Sending content " << content_stream.str() << endl;
}

void send_frowvec(SocketServer<WS>* server,
                  shared_ptr<SocketServer<WS>::Connection> connection,
                  string canvasId, string type,
                  float* vector, int n_elem)
{
  ptree content;
  content.put("action", "change_points");
  content.put("type", type);
  content.put("canvasId", canvasId);
  log_json(content);
  size_t data_size = sizeof(float) * n_elem;
  send_message(server, connection, "spectrogram", content, vector, data_size);
}

void send_spectrogram_new(SocketServer<WS>* server,
                          shared_ptr<SocketServer<WS>::Connection> connection,
                          spec_params_t spec_params, string canvasId)
{
  ptree content;
  content.put("action", "new");
  content.put("nblocks", spec_params.nblocks);
  content.put("nfreqs", spec_params.nfreqs);
  content.put("fs", spec_params.fs);
  content.put("length", spec_params.spec_len);
  content.put("canvasId", canvasId);
  log_json(content);
  send_message(server, connection, "spectrogram", content, NULL, -1);
}

void send_spectrogram_update(SocketServer<WS>* server,
                             shared_ptr<SocketServer<WS>::Connection> connection,
                             spec_params_t spec_params, string canvasId,
                             fmat& spec_mat)
{
  ptree content;
  content.put("action", "update");
  content.put("nblocks", spec_params.nblocks);
  content.put("nfreqs", spec_params.nfreqs);
  content.put("canvasId", canvasId);
  size_t data_size = sizeof(float) * spec_mat.n_elem;
  float* spec_arr = (float*) malloc(data_size);
  serialize_spec_mat(&spec_params, spec_mat, spec_arr);
  log_json(content);
  send_message(server, connection, "spectrogram", content, spec_arr, data_size);
  free(spec_arr);
}

void send_change_points(SocketServer<WS>* server,
                        shared_ptr<SocketServer<WS>::Connection> connection,
                        string canvasId,
                        cp_data_t* cp_data)
{
  send_frowvec(server, connection, canvasId, "change_points", cp_data->cp.memptr(), cp_data->cp.n_elem);
  send_frowvec(server, connection, canvasId, "summed_signal", cp_data->m.memptr(), cp_data->cp.n_elem);
}

void on_file_spectrogram(SocketServer<WS>* server, shared_ptr<SocketServer<WS>::Connection> connection, ptree data)
{
  string filename = data.get<string>("filename");
  float duration = data.get<float>("duration");

  spec_params_t spec_params;
  char *filename_c = new char[filename.length() + 1];
  strcpy(filename_c, filename.c_str());
  get_eeg_spectrogram_params(&spec_params, filename_c, duration);
  print_spec_params_t(&spec_params);
  const char* ch_name;
  for (int ch = 0; ch < NUM_CH; ch++)
  {
    ch_name = CH_NAME_MAP[ch];
    send_spectrogram_new(server, connection, spec_params, ch_name);
    fmat spec_mat = fmat(spec_params.nfreqs, spec_params.nblocks);
    eeg_spectrogram(&spec_params, ch, spec_mat);

    cp_data_t cp_data;
    init_cp_data_t(&cp_data, spec_mat.n_rows);
    get_change_points(spec_mat, &cp_data);

    send_spectrogram_update(server, connection, spec_params, ch_name, spec_mat);
    send_change_points(server, connection, ch_name, &cp_data);
    this_thread::sleep_for(chrono::seconds(5)); // TODO(joshblum): fix this..
    break;
  }
  close_edf(filename_c);
}

void receive_message(SocketServer<WS>* server, shared_ptr<SocketServer<WS>::Connection> connection, string type, ptree content)
{
  ostringstream content_stream;
  write_json(content_stream, content);
  if (type == "request_file_spectrogram")
  {
    on_file_spectrogram(server, connection, content);
  }
  else if (type == "information")
  {
    cout << content_stream.str() << endl;
  }
  else
  {
    cout << "Unknown type: " << type << " and content: " << content_stream.str() << endl;
  }
}

int main()
{
  //WebSocket (WS)-server at PORT using NUM_THREADS threads
  SocketServer<WS> server(PORT, NUM_THREADS);

  auto& ws = server.endpoint["^/compute/spectrogram/?$"];

//C++14, lambda parameters declared with auto
//For C++11 use: (shared_ptr<SocketServer<WS>::Connection> connection, shared_ptr<SocketServer<WS>::Message> message)
  ws.onmessage = [&server](auto connection, auto message)
  {
    //To receive message from client as string (data_ss.str())
    stringstream data_ss;
    message->data >> data_ss.rdbuf();
    ptree json;
    read_json(data_ss, json);
    // TODO add error checking for null fields
    string type = json.get<string>("type");
    //ptree content = json.get<ptree>("content");
    ptree content;

    receive_message(&server, connection, type, content);
  };

  ws.onopen = [](auto connection)
  {
    cout << "WebSocket opened" << endl;
  };


  //See RFC 6455 7.4.1. for status codes
  ws.onclose = [](auto connection, int status, const string & reason)
  {
    cout << "Server: Closed connection " << (size_t)connection.get() << " with status code " << status << endl;
  };

  //See http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html, Error Codes for error code meanings
  ws.onerror = [](auto connection, const boost::system::error_code & ec)
  {
    cout << "Server: Error in connection " << (size_t)connection.get() << ". " <<
         "Error: " << ec << ", error message: " << ec.message() << endl;
  };


  thread server_thread([&server]()
  {
    cout << "WebSocket Server started at port: " << PORT << endl;
    //Start WS-server
    server.start();
  });

  server_thread.join();

  return 0;
}
