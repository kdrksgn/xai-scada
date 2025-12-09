
##! Modbus Zeek Script for OntoXAI
##! Extracts Function Codes (FC) and Coil/Register addresses.

module ModbusOnto;

export {
    redef enum Log::ID += { LOG };
    
    type Info: record {
        ts: time &log;
        uid: string &log;
        id: conn_id &log;
        fc: count &log;
        func_name: string &log;
        addr: count &log &optional;
        data: string &log &optional;
    };
}

event modbus_message(c: connection, headers: ModbusHeaders, is_orig: bool)
    {
    local rec: Info = [
        $ts=network_time(),
        $uid=c$uid,
        $id=c$id,
        $fc=headers$function_code,
        $func_name=Modbus::function_codes[headers$function_code]
    ];
    
    Log::write(LOG, rec);
    }
