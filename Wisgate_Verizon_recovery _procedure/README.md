# Provisional Customer Procedure — WisGate Verizon Cellular Recovery

**Status:** Temporary workaround pending final guidance from RAKwireless  
**Date:** November 17, 2026  
**Validated on:** WisGateOS 2.2.2_ARDUINO b96, Quectel EG95, Verizon APN `vzwinternet`  
**Validated test gateway:** TestGateway

## Purpose

Use this procedure when a gateway registers on Verizon and the SIM is ready, but the cellular data interface does not come online and the logs repeatedly show:

```text
requestSetupDataCall QMUXResult = 0x1, QMUXError = 0xe
Dail Failed
```

or a manual QMI test returns:

```text
QMI protocol error (14): 'CallFailed'
verbose call end reason (2,241): [internal] interface-in-use-config-match
```

The observed condition is that the modem has an active or reserved data context, but the gateway connection manager cannot reuse it. The temporary workaround disables automatic MBN selection, deactivates the selected carrier profile, resets the modem, and allows the host to establish a new QMI session.

## Important limits

- Apply this procedure to **one test gateway at a time**.
- Do **not** deploy it fleet-wide until RAKwireless confirms the permanent resolution and supported firmware/configuration.
- Keep Wi-Fi or another backup connection working throughout the procedure.
- The MBN configuration change is persistent across reboots.
- The modem will restart and cellular service will be briefly unavailable.
- Do not paste terminal prompts or previous command output back into the shell. Copy only the command lines shown below.

## Prerequisites

- Root SSH access to the gateway.
- A working Wi-Fi failover or local maintenance connection.
- A maintenance window of approximately 10–15 minutes.
- Verizon APN confirmed as `vzwinternet`.
- Record the gateway name, serial number, modem firmware, SIM ICCID, and current time before beginning.

## 1. Record the current state

Run these commands from the Linux shell:

```sh
ifstatus wwan
ip addr show dev wwan0
ip route
ps w | grep -E '[q]uectel|[q]micli|[M]odemManager|[q]mi-proxy'
logread | grep -E 'quectel-CM|requestSetupDataCall|Dial Failed|Dail Failed|IPv4ConnectionStatus' | tail -n 40
```

Save the output for the support case.

## 2. Stop all cellular dial-up tools

```sh
/etc/init.d/quectel-CM stop
ps w | grep -E '[q]uectel|[q]micli|[M]odemManager|[q]mi-proxy'
```

The second command should return no cellular connection-manager process.

If the stop command reports `Command failed: Not found`, continue only if the process check is empty. This can mean the service is already stopped.

## 3. Open the modem AT-command port

First confirm the available ports:

```sh
ls -l /dev/ttyATWWAN /dev/ttyUSB* 2>/dev/null
```

Open the port used on the validated unit:

```sh
microcom -s 115200 /dev/ttyUSB2
```

If `/dev/ttyATWWAN` exists and points to the AT port, it may be used instead:

```sh
microcom -s 115200 /dev/ttyATWWAN
```

At the blank `microcom` screen, type `AT` and press Enter. The modem should answer `OK`.

## 4. Capture the modem state before changing it

Run these commands inside `microcom`, one line at a time:

```text
AT
AT+QMBNCFG="List"
AT+QMBNCFG="AutoSel"
AT+CGACT?
AT+CGPADDR=1
```

Save the complete responses. Before the workaround, the test unit showed the Verizon MBN selected and active and `AutoSel` set to `1`.

## 5. Apply the temporary workaround

Still inside `microcom`, enter the following commands one at a time and wait for `OK` after each command:

```text
AT+QMBNCFG="AutoSel",0
AT+QMBNCFG="Deactivate"
AT+CFUN=1,1
```

Expected behavior:

- Each command returns `OK`.
- `AT+CFUN=1,1` restarts the modem.
- The AT session may close and the USB modem devices may disappear briefly. This is normal.

If any command returns `ERROR`, stop the procedure, do not repeat commands, and follow the escalation section below.

## 6. Verify the modem after restart

Wait until `/dev/ttyUSB2` and `/dev/cdc-wdm0` return, then make sure no dial-up tool has restarted:

```sh
ps w | grep -E '[q]uectel|[q]micli|[M]odemManager|[q]mi-proxy'
```

If `quectel` is running, stop it again:

```sh
/etc/init.d/quectel-CM stop
```

Open the AT port again:

```sh
microcom -s 115200 /dev/ttyUSB2
```

Run:

```text
AT+QMBNCFG="List"
AT+QMBNCFG="AutoSel"
AT+CGACT?
AT+CGPADDR=1
```

Expected results:

- `AT+QMBNCFG="AutoSel"` returns `0`.
- No MBN entry is shown as both selected and active.
- The modem may show an active PDP context and an assigned carrier IP address; this alone does not prove the Linux interface is working.

Exit `microcom` before proceeding. The common exit sequence is **Ctrl+X**.

## 7. Perform a controlled QMI connection test

With all other dial-up tools stopped, run:

```sh
qmicli -d /dev/cdc-wdm0 --wds-start-network="apn=vzwinternet,ip-type=4" --wds-follow-network
```

A successful test displays:

```text
[/dev/cdc-wdm0] Network started
        Packet data handle: '<number>'
```

Press **Ctrl+C** to end the controlled test. This stops the test connection cleanly.

If error `(2,241) interface-in-use-config-match` remains, do not repeatedly retry. Capture the complete output and escalate to RAKwireless.

## 8. Return control to the gateway

Restart the normal gateway connection manager:

```sh
/etc/init.d/quectel-CM start
ps w | grep '[q]uectel'
```

Allow two to three minutes for registration and DHCP, then verify:

```sh
ifstatus wwan
ip addr show dev wwan0
ip route
logread | grep -E 'quectel-CM|requestSetupDataCall|Dial Failed|Dail Failed|IPv4ConnectionStatus' | tail -n 40
ping -I wwan0 -c 4 8.8.8.8
```

Also check WisDM under **Gateway → Network interface → WAN**. The expected state is:

- Cellular: **Active / Primary**
- Wi-Fi: **Secondary**
- No repeating `Dail Failed` messages in the new logs

## Success criteria

The workaround is successful only when all applicable checks pass:

- The controlled `qmicli` test reports `Network started`.
- Error `(2,241) interface-in-use-config-match` no longer occurs.
- `ifstatus wwan` reports the interface up.
- `wwan0` receives an IPv4 address.
- A route is installed for cellular or the gateway's failover manager reports cellular active.
- `ping -I wwan0` succeeds.
- WisDM reports Cellular as Active / Primary.

## If the procedure does not succeed

Keep Wi-Fi connected and collect the following before contacting support:

```sh
ifstatus wwan
ip addr show dev wwan0
ip route
ps w | grep -E '[q]uectel|[q]micli|[M]odemManager|[q]mi-proxy'
qmicli -d /dev/cdc-wdm0 --wds-get-packet-service-status
qmicli -d /dev/cdc-wdm0 --wda-get-data-format
logread | grep -E 'quectel-CM|requestSetupDataCall|Dial Failed|Dail Failed|IPv4ConnectionStatus' | tail -n 100
```

Also provide:

- The before-and-after responses to `AT+QMBNCFG="List"` and `AT+QMBNCFG="AutoSel"`.
- The complete output from the controlled `qmicli` test.
- Gateway model and WisGateOS version.
- Quectel modem firmware version.
- Approximate time of the test and confirmation that Wi-Fi remained available.
- Remote access details, using the organization's approved secure method, if requested by RAKwireless.

## Provisional rollback

Use this rollback only if directed by the product team or RAKwireless, because it re-enables automatic MBN selection and may restore the original failure condition.

Stop the connection manager, open the AT port, and run:

```text
AT+QMBNCFG="AutoSel",1
AT+CFUN=1,1
```

After the modem restarts, restore the gateway service:

```sh
/etc/init.d/quectel-CM start
```

Then repeat the verification commands in Section 8.

## Current engineering conclusion

On the test gateway, disabling MBN automatic selection and deactivating the selected Verizon MBN cleared the QMI session conflict. A manual IPv4 QMI connection then started successfully, and WisDM subsequently showed Cellular as the active primary WAN with Wi-Fi as secondary. This confirms the workaround on that unit, but it does not yet establish the root cause or authorize fleet-wide deployment.

Related Quectel community case supplied during the investigation:  
https://forums.quectel.com/t/ec25-af-w-verizon-call-end-reason-verbose-is-241/8334
