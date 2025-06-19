import { Link } from "wouter";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { applications } from "@/data/applications";

const getCategoryColor = (category: string) => {
  switch (category) {
    case 'Signal Processing': return 'bg-blue-100 text-blue-800';
    case 'Equipment & Technical': return 'bg-green-100 text-green-800';
    case 'Station Operations': return 'bg-purple-100 text-purple-800';
    case 'Emergency Communications': return 'bg-red-100 text-red-800';
    case 'Advanced Applications': return 'bg-orange-100 text-orange-800';
    default: return 'bg-gray-100 text-gray-800';
  }
};

export default function MatrixTable() {
  return (
    <Card className="overflow-hidden">
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-800 hover:bg-slate-800">
              <TableHead className="text-white font-semibold">Application Area</TableHead>
              <TableHead className="text-white font-semibold">Practical Example/Tool</TableHead>
              <TableHead className="text-white font-semibold">Description</TableHead>
              <TableHead className="text-white font-semibold">Source(s)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {applications.map((app) => (
              <TableRow key={app.slug} className="hover:bg-gray-50">
                <TableCell className="font-medium">
                  <Link href={`/application/${app.slug}`}>
                    <div className="cursor-pointer">
                      <Badge className={`mb-2 ${getCategoryColor(app.category)}`} variant="secondary">
                        {app.category}
                      </Badge>
                      <div className="text-gov-navy hover:text-gov-blue hover:underline">
                        <strong>{app.area}</strong>
                      </div>
                    </div>
                  </Link>
                </TableCell>
                <TableCell className="text-gray-900">{app.example}</TableCell>
                <TableCell className="text-gray-700">{app.description}</TableCell>
                <TableCell className="text-gov-blue text-sm">{app.sources}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Card>
  );
}
