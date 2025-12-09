
##! EtherNet/IP (ENIP) Zeek Script for OntoXAI

module ENIPOnto;

export {
    redef enum Log::ID += { LOG };
}

event enip_register_session(c: connection, header: ENIP_Header)
    {
    # Log session registration
    }
