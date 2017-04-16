
#include <obliv.h>
#include "yao.h"

int connectTcpOrDie(ProtocolDesc* pd, const char* remote, const char* port) {
  remote = (strcmp(remote, "--")) ? remote : NULL;
  // if I have a remote host defined, i'm the client, otherwise the server
  int party_id = (remote) ? 2 : 1;

  if (remote) {
    // we are the TCP client
    if (protocolConnectTcp2P(pd, remote, port)) {
      fprintf(stderr, "TCP Connect Fail\n");
      exit(1);
    }
  } else {
    // we are the TCP server
    if (protocolAcceptTcp2P(pd, port)) {
      fprintf(stderr, "TCP Accept Fail\n");
      exit(1);
    }
  }

  setCurrentParty(pd, party_id);

  return party_id;
}
